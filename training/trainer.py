"""
Train HOTFormerLoc or MinkLoc3Dv2 model, with support for metric localisation.
Based on MinkLoc3Dv2 training script by Jacek Komorowski.

Ethan Griffiths (Data61, Pullenvale)
"""

import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import tqdm
import pathlib
import typing as tp
import wandb
import submitit
import matplotlib.pyplot as plt
import seaborn as sns
from ocnn.octree import Points
from timm.utils import ModelEmaV3
from timm.optim.lamb import Lamb
import wandb_osh
from wandb_osh.hooks import TriggerWandbSyncHook

from misc.utils import TrainingParams, get_datetime, update_params_from_dict
from misc.torch_utils import release_cuda, set_seed, to_device
from misc.logger import Logger
from models.losses.loss import make_losses, kdloss
from models.losses.loss_utils import metrics_mean
from models.model_factory import model_factory
from models.hotformerloc import HOTFormerLoc
from models.hotformerloc_metric_loc import HOTFormerMetricLoc
from models.octree import OctreeT, get_octant_centroids_from_points
from dataset.dataset_utils import make_dataloaders
from eval.evaluate_metric_loc_splits_sgv import evaluate, print_eval_stats, write_eval_stats, EVAL_MODES
from eval.vis_utils import remove_rt_attn_padding, rowwise_cosine_sim, off_diagonal, \
    colourise_points_by_height, colourise_points_by_similarity, \
        create_heatmap

os.environ["WANDB__SERVICE_WAIT"] = "300"  # prevent crash if wandb is slow
WANDB_OFFLINE = False  # Use wandb in offline mode with sync hooks running on login node
wandb_osh.set_log_level("ERROR")

class NetworkTrainer:
    """
    Class to handle main training loop for HOTFormerLoc. Supports checkpointing
    and automatic resubmission to SLURM clusters through submitit.

    Note that this implementation is a little rushed and may have some unforseen
    quirks.
    """

    def __init__(self):
        self.model = None
        self.model_ema = None
        self.optimizer = None
        self.scheduler = None
        self.params = None
        self.model_pathname = None
        self.device = None
        self.wandb_id = None
        self.resume = False
        self.start_epoch = 1
        self.curr_epoch = 1
        self.count_batches = 0
        self.best_avg_AR_1 = 0.0
        self.checkpoint_extension = '_latest.ckpt'
        
    def __call__(
        self,
        params: TrainingParams = None,
        *args,
        **kwargs,
    ):
        """
        Handle logic for starting/resuming training when called manually or by
        submitit.
        """
        self.params = params
        checkpoint_path = kwargs.get('checkpoint_path')
        self.resume = checkpoint_path is not None
        if self.params.finetune and not self.resume:
            raise ValueError('Cannot fine-tune without specifying weights (use `--resume_from`)')

        # Set params for hyperparam search
        if not self.resume:
            if self.params.hyperparam_search:
                if len(args) == 1 and isinstance(args[0], dict):  # This is required for submitit job arrays currently
                    kwargs = args[0]
                assert kwargs != {}, 'No valid hyperparams were provided for search'
                self.params = update_params_from_dict(self.params, kwargs)
        self.params.print()
        print(f'OMP NUM THREADS: {os.getenv("OMP_NUM_THREADS")}')
        
        # Seed RNG
        set_seed()
        print('Determinism: Enabled')

        # Initialise model, optimiser, and scheduler 
        self.init_model_optim_sched()
        
        # Load state from ckpt if resubmitted, otherwise start from scratch
        if self.resume:
            self.model_pathname = checkpoint_path.split(self.checkpoint_extension)[0]
            state = torch.load(checkpoint_path)
            if self.params.finetune:  # Finetuning
                if os.path.splitext(checkpoint_path)[1] == '.ckpt':
                    state = state['model_state_dict']
                try:
                    self.model.load_state_dict(state)
                except RuntimeError:
                    # Check if HOTFormerLoc weights are being loaded for HOTFormerMetricLoc
                    if isinstance(self.model, HOTFormerMetricLoc):
                        self.model.hotformerloc_global.load_state_dict(state)
                    else:
                        raise
                # Need to re-init model_ema to checkpoint weights
                if self.model_ema is not None:
                    self.init_model_ema()
                print(f'Begin fine-tuning of {self.model_pathname}')

            else:  # Resume training
                try:
                    self.start_epoch = state['epoch']
                    self.curr_epoch = self.start_epoch
                    self.wandb_id = state['wandb_id']
                    self.best_avg_AR_1 = state['best_avg_AR_1']
                    self.model.load_state_dict(state['model_state_dict'])
                    self.optimizer.load_state_dict(state['optim_state_dict'])
                    if self.scheduler is not None:
                        self.scheduler.load_state_dict(state['sched_state_dict'])
                    if self.model_ema is not None:
                        self.model_ema.load_state_dict(state['model_ema_state_dict'])
                except KeyError:
                    error_msg = (
                        "Invalid checkpoint file provided. Only files ending "
                        "in '.ckpt' are valid for resuming training."
                    )
                    raise ValueError(error_msg)
                print(f'Resuming training of {self.model_pathname} from epoch {self.start_epoch}')

        if (not self.resume) or self.params.finetune:
            # Create model class
            model_name = self.params.model_params.model
            if self.params.finetune:
                # Save new model_pathname (combining weights name and model name)
                orig_model_name = os.path.splitext(os.path.basename(self.model_pathname))[0]
                model_name += f'--finetune--{orig_model_name}'
            s = get_datetime()
            model_name += f'_{s}'
            # Add SLURM job ID to prevent overwriting paths for jobs running at same time
            if 'SLURM_JOB_ID' in os.environ:
                model_name += f"_job{os.environ['SLURM_JOB_ID']}"
            weights_path = self.create_weights_folder(self.params.dataset_name)
            self.model_pathname = os.path.join(weights_path, model_name)
            print('Model name: {}'.format(model_name))
            
        if hasattr(self.model, 'print_info'):
            self.model.print_info()
        else:
            n_params = sum([param.nelement() for param in self.model.parameters()])
            print('Number of model parameters: {}'.format(n_params))
                    
        # Initialise the logger
        logging_level = 'DEBUG' if self.params.verbose else 'INFO'
        self.logger = Logger(log_file=None, logging_level=logging_level, local_rank=-1)

        # Begin train loop
        msg = f'Begin training on {os.environ['HOSTNAME']}'
        if 'SLURM_JOB_ID' in os.environ:
            msg += f" with job ID {os.environ['SLURM_JOB_ID']}"
        self.logger.info(msg)
        self.do_train()

    def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> submitit.helpers.DelayedSubmission:
        """
        This function is called asynchronously by submitit when a job is timed
        out. Dumps the current model state to disk and returns a
        DelayedSubmission to resubmit the job.
        """
        if WANDB_OFFLINE:
            self.trigger_sync()
        checkpoint_path = self.model_pathname + self.checkpoint_extension
        print(f'Training interupted at epoch {self.curr_epoch}. '
              f'Saving ckpt to {checkpoint_path} and resubmitting.')
        if not os.path.exists(checkpoint_path):
            self.save_checkpoint(checkpoint_path)
        training_callable = NetworkTrainer()
        delayed_submission = submitit.helpers.DelayedSubmission(
            training_callable,
            self.params,
            checkpoint_path=checkpoint_path,
        )
        return delayed_submission

    def save_checkpoint(self, checkpoint_path: str):
        # Save checkpoint of training state (as opposed to just model weights)
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        state = {
            'epoch': self.curr_epoch,
            'wandb_id': self.wandb_id,
            'best_avg_AR_1': self.best_avg_AR_1,
            'model_state_dict': self.model.state_dict(),                
            'optim_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state['sched_state_dict'] = self.scheduler.state_dict()
        if self.model_ema is not None:
            state['model_ema_state_dict'] = self.model_ema.state_dict()
        torch.save(state, checkpoint_path)

    def init_model_ema(self):
        self.model_ema = ModelEmaV3(self.model, decay=0.9998)
    
    def init_model_optim_sched(self):
        """
        Initialise the model, optimiser, and scheduler from the provided parameters.
        """
        self.model = model_factory(self.params)

        # Move the model to the proper device before configuring the optimizer
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        print('Model device: {}'.format(self.device))

        # Setup exponential moving average of model weights, SWA could be used here too
        if self.params.mesa > 0.0:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.init_model_ema()

        # Training elements
        if self.params.optimizer == 'Adam':
            optimizer_fn = torch.optim.Adam
        elif self.params.optimizer == 'AdamW':
            optimizer_fn = torch.optim.AdamW
        elif self.params.optimizer == 'Lamb':
            optimizer_fn = Lamb
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.params.optimizer}")

        if self.params.weight_decay is None or self.params.weight_decay == 0:
            self.optimizer = optimizer_fn(self.model.parameters(), lr=self.params.lr)
        else:
            self.optimizer = optimizer_fn(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        if self.params.scheduler is not None:
            if self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.epochs+1,
                                                                    eta_min=self.params.min_lr)
            elif self.params.scheduler == 'MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.params.scheduler_milestones, gamma=self.params.gamma)
            elif self.params.scheduler == 'ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.params.gamma)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.params.scheduler))

        if self.params.warmup_epochs is not None:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup_scheduler, self.scheduler], [self.params.warmup_epochs])

    def warmup(self, epoch: int):
        # Linear scaling lr warmup
        min_factor = 1e-3
        return max(float(epoch / self.params.warmup_epochs), min_factor)

    def tensors_to_numbers(self, stats):
        stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
        return stats

    def print_global_stats(self, phase, stats):
        s = f"{phase}:  total loss: {stats['loss_total']:.4f}   PR loss: {stats['loss']:.4f}   "
        if 'loss_mesa' in stats:
            s += f"MESA loss: {stats['loss_mesa']:.4f}   "
        if 'local_qkv_std_loss' in stats:
            s += f"local qkv std loss: {stats['local_qkv_std_loss']:.4f}   "
        if 'rt_qkv_std_loss' in stats:
            s += f"rt qkv std loss: {stats['rt_qkv_std_loss']:.4f}   "
        if 'qkv_weight_norm_loss' in stats:
            s += f"qkv weight norm loss: {stats['qkv_weight_norm_loss']:.4f}   "
        s += f"embedding norm: {stats['avg_embedding_norm']:.3f}   "
        if 'num_triplets' in stats:
            s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
                f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
        if 'positives_per_query' in stats:
            s += f"#positives per query: {stats['positives_per_query']:.1f}   "
        if 'best_positive_ranking' in stats:
            s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
        if 'recall' in stats:
            s += f"Recall@1: {stats['recall'][1]:.4f}   "
        if 'ap' in stats:
            s += f"AP: {stats['ap']:.4f}   "
        self.logger.info(s)

    def print_local_stats(self, phase, stats):
        s = f"{phase}:  local loss: {stats['loss']:.4f}   "
        if 'coarse_loss' in stats:
            s += f"coarse loss: {stats['coarse_loss']:.4f}   "
        if 'fine_loss' in stats:
            s += f"fine loss: {stats['fine_loss']:.4f}   "
        if 'PIR' in stats:
            s += f"PIR: {stats['PIR']:.4f}   "
        if 'IR' in stats:
            s += f"IR: {stats['IR']:.4f}   "
        if 'RRE' in stats:
            s += f"RRE: {stats['RRE']:.4f}   "
        if 'RTE' in stats:
            s += f"RTE: {stats['RTE']:.4f}   "
        if 'RR' in stats:
            s += f"RR: {stats['RR']:.4f}   "
        self.logger.info(s)

    def print_stats(self, phase, stats):
        self.print_global_stats(phase, stats['global'])
        if self.params.local.enable_local and 'local' in stats:
            self.print_local_stats(phase, stats['local'])
    
    def create_weights_folder(self, dataset_name : str):
        # Create a folder to save weights of trained models
        this_file_path = pathlib.Path(__file__).parent.absolute()
        temp, _ = os.path.split(this_file_path)
        weights_path = os.path.join(temp, 'weights', dataset_name)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
        return weights_path

    def log_stage_gradient_magnitudes(self, octformer_variant: bool):
        """
        Compute the gradient magnitudes for each stage of the model, to determine if
        gradients are vanishing/exploding.
        """
        # NOTE: This doesn't capture every layer in each stage, but should give a
        # reasonable insight into what's happening each stage.
        if octformer_variant:
            stages = self.model.backbone.backbone.layers            
        else:  # MinkLoc
            stages = self.model.backbone.blocks

        stage_grad_mags = {}
        for stage_i, stage in enumerate(stages):
            grad_mags_list = []
            for param in stage.parameters():
                if param.grad is None:
                    continue
                grad_mags_list.append(param.grad.abs().mean().item())
            stage_grad_mags[stage_i] = np.mean(grad_mags_list)
        return stage_grad_mags

    def log_eval_stats(self, global_metrics: dict, local_metrics: dict):
        eval_stats = {}
        for database_name in global_metrics.keys():
            for split in global_metrics[database_name].keys():
                split_log_key = split
                if split == 'average':
                    split_log_key = database_name
                eval_stats[split_log_key] = {
                    'recall@1': {},
                    'recall@5': {},
                    'recall@20': {},
                    'recall@1%': {},
                    'MRR': {},
                }
                if 'Re-Ranking' in EVAL_MODES:
                    eval_stats[split_log_key].update({
                        'recall@1_rerank': {},
                        'recall@5_rerank': {},
                        'recall@20_rerank': {},
                        'recall@1%_rerank': {},
                        'MRR_rerank': {},
                    })

                for radius in self.params.eval_radius:
                    eval_stats[split_log_key]['recall@1'][radius] = global_metrics[database_name][split]['recall'][radius][0]
                    eval_stats[split_log_key]['recall@5'][radius] = global_metrics[database_name][split]['recall'][radius][5-1]
                    eval_stats[split_log_key]['recall@20'][radius] = global_metrics[database_name][split]['recall'][radius][20-1]
                    eval_stats[split_log_key]['recall@1%'][radius] = global_metrics[database_name][split]['recall@1%'][radius]
                    eval_stats[split_log_key]['MRR'][radius] = global_metrics[database_name][split]['MRR'][radius]
                    if 'Re-Ranking' in EVAL_MODES:
                        eval_stats[split_log_key]['recall@1_rerank'][radius] = global_metrics[database_name][split]['recall_rr'][radius][0]
                        eval_stats[split_log_key]['recall@5_rerank'][radius] = global_metrics[database_name][split]['recall_rr'][radius][5-1]
                        eval_stats[split_log_key]['recall@20_rerank'][radius] = global_metrics[database_name][split]['recall_rr'][radius][20-1]
                        eval_stats[split_log_key]['recall@1%_rerank'][radius] = global_metrics[database_name][split]['recall@1%_rr'][radius]
                        eval_stats[split_log_key]['MRR_rerank'][radius] = global_metrics[database_name][split]['MRR_rr'][radius]

                if len(local_metrics) == 0:
                    continue
                for eval_mode in EVAL_MODES:
                    if eval_mode not in local_metrics[database_name][split]:
                        continue
                    eval_stats[split_log_key][eval_mode] = {}
                    for metric in local_metrics[database_name][split][eval_mode].keys():
                        if 'failure' in metric:  # ignore failure indices, no need to log to wandb
                            continue
                        eval_stats[split_log_key][eval_mode][metric] = local_metrics[database_name][split][eval_mode][metric]
        return eval_stats

    def log_feats(self, feature_maps: tp.List) -> tp.Dict:
        """
        Logs feature maps coming from MinkLoc.
        """
        tic = time.time()
        BATCH_IDX = 0  # only look at one batch element
        # Only log once per epoch
        if self.count_batches > 1:
            return {}

        # Make absolutely sure that gradients aren't touched here
        with torch.no_grad():
            stats = {'local_token_unique_sim': {}, 'local_token_sim_matrix': {},
                     'pointcloud': {}, 'pca_variance': {}}
            # Log query and positive (queries and positives are neighbour pairs within the batch)
            assert BATCH_IDX % 2 == 0, "Queries are even indices, positives are odd indices"
            query_pos_indices = [BATCH_IDX, BATCH_IDX + 1]
            for i, batch_idx in enumerate(query_pos_indices):
                pcl_type = 'query' if i == 0 else 'positive'
                # Log the point cloud itself (in this case using the quantized point cloud)
                pcl_orig = release_cuda(feature_maps[0].coordinates_at(batch_index=batch_idx), to_numpy=True)
                pcl_orig_colours = colourise_points_by_height(pcl_orig) * 255.0
                pcl_orig_combined = np.concatenate([pcl_orig, pcl_orig_colours], axis=1)
                stats['pointcloud'][f'{pcl_type}_depth_orig'] = wandb.Object3D(pcl_orig_combined)

                # Log embedding similarities and colourised point cloud for each layer
                for j, feats_j in enumerate(feature_maps):
                    pcl_j, feature_map_j = feats_j.coordinates_and_features_at(batch_index=batch_idx)
                    pcl_j, feature_map_j = release_cuda(pcl_j, to_numpy=True), release_cuda(feature_map_j)
                    if i == 0:  # currently only logging similarity for one point cloud
                        temp_sim = rowwise_cosine_sim(feature_map_j, feature_map_j)
                        stats['local_token_sim_matrix'][f'layer_{j}'] = wandb.Image(create_heatmap(temp_sim, title=f'Layer {j}'))
                        plt.close()
                        # Collect unique values of token similarity (off diagonal)
                        stats['local_token_unique_sim'][f'layer_{j}'] = wandb.Histogram(off_diagonal(temp_sim).numpy())
                    # Plot points colourised by PCA embeddings
                    pcl_j_colours, feature_map_j_variance = colourise_points_by_similarity(
                        feature_map_j.numpy(), mode='pca', return_explained_variance=True
                    )
                    pcl_j_colours *= 255.0
                    pcl_j_combined = np.concatenate([pcl_j, pcl_j_colours], axis=1)
                    stats['pointcloud'][f'{pcl_type}_layer_{j}'] = wandb.Object3D(pcl_j_combined)
                    stats['pca_variance'][f'{pcl_type}_layer_{j}'] = wandb.Histogram(feature_map_j_variance)
                
        self.logger.debug(f'Logged feats in {time.time() - tic:.4f}s')
        return stats

    def log_feats_and_attn_maps(self, feats_and_attn_maps: tp.List, octree: OctreeT,
                                points: Points, softmax=False) -> tp.Dict:
        """
        Logs various things including attention maps, average token similarity.
        """
        tic = time.time()
        VIZ_BLOCKS = 3
        VIZ_HEADS = 2
        VIZ_LOCAL_WINDOWS = 1
        BATCH_IDX = 0  # just take samples from the first batch element for attn maps
        EPS = 100  # epsilon for filtering masked tokens (as masked tokens may have a slightly different value after attention)
        MIN_ATTN_VALUE = octree.invalid_mask_value + EPS
        # Only log once per epoch
        if self.count_batches > 1:
            return {}
        
        def get_rt_heatmap_ticklabels() -> tp.List:
            """
            Create ticklabels for relay token heatmaps to indicate where each
            octree level starts/ends.
            """
            if octree.num_pyramid_levels == 0:  # OctFormer
                return []
            num_rt = octree.batch_num_relay_tokens_combined[BATCH_IDX]
            rt_idx_per_depth_list = []
            for depth_j in octree.pyramid_depths:
                rt_idx_per_depth_list.append(octree.batch_num_windows[depth_j][BATCH_IDX].item())
            ticklabels = ['' for _ in range(num_rt)]
            num_rt_depth_cumsum = np.cumsum(rt_idx_per_depth_list)
            for num_rt_depth_j in num_rt_depth_cumsum[:-1]:
                ticklabels[num_rt_depth_j] = num_rt_depth_j
            return ticklabels
        def log_hotformer() -> tp.Dict:
            """
            Log feats and attn maps for HOTFormerLoc variants.
            """
            # Make absolutely sure that gradients aren't touched here
            with torch.no_grad():
                stats = {'rt_attn_map': {}, 'local_attn_map': {}, 'local_rpe': {},
                        'rt_token_unique_sim': {}, 'local_token_unique_sim': {},
                        'rt_token_sim_matrix': {}, 'local_token_sim_matrix': {},
                        'pointcloud': {}, 'pca_variance': {}}
                block_indices = np.unique(np.linspace(0, len(feats_and_attn_maps)-1,
                                          VIZ_BLOCKS, dtype=np.int32)).tolist()
                rt_ticklabels = get_rt_heatmap_ticklabels()
                for block_idx in block_indices:
                    if 'rt_attn' in feats_and_attn_maps[block_idx]:
                        rt_attn_maps_i = feats_and_attn_maps[block_idx]['rt_attn']['attn_map']
                        num_heads = rt_attn_maps_i.size(dim=1)
                        head_indices = np.unique(np.linspace(0, num_heads-1, VIZ_HEADS,
                                                 dtype=np.int32)).tolist()
                        for k, head_kdx in enumerate(head_indices):
                            temp_attn_map = rt_attn_maps_i[BATCH_IDX, head_kdx].to(self.device)  # move to GPU as octree is on GPU
                            temp_attn_map = remove_rt_attn_padding(temp_attn_map, octree, BATCH_IDX)
                            if softmax:
                                temp_attn_map = F.softmax(temp_attn_map, dim=-1)
                            stats['rt_attn_map'][f'block_{block_idx}_head_{k}'] \
                                = wandb.Image(create_heatmap(temp_attn_map.cpu(), rt_ticklabels, min_value=MIN_ATTN_VALUE, 
                                                             title=f'Block {block_idx} - Head {k}'))
                            plt.close()  # close fig to prevent going OOM
                    if 'local_attn' in feats_and_attn_maps[block_idx]:
                        local_attn_maps_i = feats_and_attn_maps[block_idx]['local_attn']
                        for j, depth_j in enumerate(local_attn_maps_i.keys()):
                            local_rpe_i_depth_j = None
                            if block_idx == 0:
                                stats['local_attn_map'][f'stage_{j}'] = {}
                                stats['local_rpe'][f'stage_{j}'] = {}
                            local_attn_maps_i_depth_j = local_attn_maps_i[depth_j]['attn_map']
                            # Also log RPE to compare differences
                            if 'rpe' in local_attn_maps_i[depth_j]:
                                local_rpe_i_depth_j = local_attn_maps_i[depth_j]['rpe']
                            num_heads = local_attn_maps_i_depth_j.size(1)
                            # Ensure window indices only index selected batch element, not all elements
                            window_start_idx = 0 if BATCH_IDX == 0 else octree.batch_boundary[depth_j][BATCH_IDX-1].item()
                            window_end_idx = octree.batch_boundary[depth_j][BATCH_IDX].item()
                            window_indices = np.unique(np.linspace(window_start_idx, window_end_idx-1,
                                                    VIZ_LOCAL_WINDOWS, endpoint=False,  # last window may contain padding, so endpoint needs to be False
                                                    dtype=np.int32)).tolist() 
                            head_indices = np.unique(np.linspace(0, num_heads-1,
                                                    VIZ_HEADS, dtype=np.int32)).tolist()
                            for k, head_kdx in enumerate(head_indices):
                                for w, window_wdx in enumerate(window_indices):
                                    temp_attn_map = local_attn_maps_i_depth_j[window_wdx, head_kdx]
                                    if softmax:
                                        temp_attn_map = F.softmax(temp_attn_map, dim=-1)
                                    stats['local_attn_map'][f'stage_{j}'][f'block_{block_idx}_head_{k}_window_{w}'] \
                                        = wandb.Image(create_heatmap(temp_attn_map, min_value=MIN_ATTN_VALUE,
                                                                     title=f'Block {block_idx} - Head {k} - Window {w}'))
                                    plt.close()  # close fig to prevent going OOM
                                    if local_rpe_i_depth_j is not None:
                                        temp_rpe = local_rpe_i_depth_j[window_wdx, head_kdx]
                                        stats['local_rpe'][f'stage_{j}'][f'block_{block_idx}_head_{k}_window_{w}'] \
                                            = wandb.Image(create_heatmap(temp_rpe, min_value=MIN_ATTN_VALUE,
                                                                         title=f'Block {block_idx} - Head {k} - Window {w}')) 
                                        plt.close()
                    # Compute similarity metrics of tokens
                    if 'rt_feats_pre_local' in feats_and_attn_maps[block_idx]:
                        rt_feats_i = feats_and_attn_maps[block_idx]['rt_feats_pre_local']
                        rt_feats_i_batch_list = []
                        # Compute similarity between all RTs
                        for j, depth_j in enumerate(rt_feats_i.keys()):
                            # Select only tokens from a single batch
                            token_batch_mask_depth_j = octree.ct_batch_idx[depth_j] == BATCH_IDX
                            rt_feats_i_batch_list.append(rt_feats_i[depth_j].to(self.device)[token_batch_mask_depth_j])
                            ############ if block_idx == 0:  # NOTE: THIS BLOCK COMPUTES RT SIMILARITY FOR EACH DEPTH SEPARATELY
                            ############     stats['rt_token_sim_matrix'][f'stage_{j}'] = {}
                            ############     stats['rt_token_unique_sim'][f'stage_{j}'] = {}
                            ############ # Select only tokens from a single batch
                            ############ token_batch_mask_depth_j = octree.ct_batch_idx[depth_j] == BATCH_IDX
                            ############ rt_feats_i_depth_j = rt_feats_i[depth_j].to(self.device)[token_batch_mask_depth_j]
                            ############ temp_sim = rowwise_cosine_sim(rt_feats_i_depth_j, rt_feats_i_depth_j).cpu()
                            ############ stats['rt_token_sim_matrix'][f'stage_{j}'][f'block_{block_idx}'] = wandb.Image(create_heatmap(temp_sim))
                            ############ plt.close()
                            ############ # Collect unique values of token similarity (off diagonal)
                            ############ stats['rt_token_unique_sim'][f'stage_{j}'][f'block_{block_idx}'] = wandb.Histogram(off_diagonal(temp_sim).numpy())
                        rt_feats_i_batch = torch.cat(rt_feats_i_batch_list, dim=0)
                        temp_sim = rowwise_cosine_sim(rt_feats_i_batch, rt_feats_i_batch).cpu()
                        stats['rt_token_sim_matrix'][f'block_{block_idx}'] = wandb.Image(create_heatmap(temp_sim, rt_ticklabels,
                                                                                                        title=f'Block {block_idx}'))
                        plt.close()
                        # Collect unique values of token similarity (off diagonal)
                        stats['rt_token_unique_sim'][f'block_{block_idx}'] = wandb.Histogram(off_diagonal(temp_sim).numpy())
                    if 'local_feats' in feats_and_attn_maps[block_idx]:
                        local_feats_i = feats_and_attn_maps[block_idx]['local_feats']
                        for j, depth_j in enumerate(local_feats_i.keys()):
                            if block_idx == 0:
                                stats['local_token_sim_matrix'][f'stage_{j}'] = {}
                                stats['local_token_unique_sim'][f'stage_{j}'] = {}
                            # Select only tokens from a single batch
                            token_batch_mask_depth_j = octree.batch_id(depth_j, octree.nempty) == BATCH_IDX
                            local_feats_i_depth_j = local_feats_i[depth_j].to(self.device)[token_batch_mask_depth_j]
                            temp_sim = rowwise_cosine_sim(local_feats_i_depth_j, local_feats_i_depth_j).cpu()
                            stats['local_token_sim_matrix'][f'stage_{j}'][f'block_{block_idx}'] = wandb.Image(create_heatmap(temp_sim,
                                                                                                                             title=f'Block {block_idx}'))
                            plt.close()
                            # Collect unique values of token similarity (off diagonal)
                            stats['local_token_unique_sim'][f'stage_{j}'][f'block_{block_idx}'] = wandb.Histogram(off_diagonal(temp_sim).numpy())

                # Log the point cloud itself
                octree_local = release_cuda(octree)
                octree_points_local = release_cuda(points)
                pcl_max_depth = get_octant_centroids_from_points(
                    octree_points_local, depth=self.params.octree_depth, quantizer=self.params.model_params.quantizer
                )
                # Log query and positive (queries and positives are neighbour pairs within the batch)
                assert BATCH_IDX % 2 == 0, "Queries are even indices, positives are odd indices"
                query_pos_indices = [BATCH_IDX, BATCH_IDX + 1]
                for i, batch_idx in enumerate(query_pos_indices):
                    pcl_type = 'query' if i == 0 else 'positive'
                    batch_mask_max_depth = octree_local.batch_id(self.params.octree_depth, octree_local.nempty) == batch_idx
                    pcl_max_depth_masked = pcl_max_depth[batch_mask_max_depth].numpy()
                    pcl_max_depth_colours = colourise_points_by_height(pcl_max_depth_masked) * 255.0
                    pcl_max_depth_combined = np.concatenate([pcl_max_depth_masked, pcl_max_depth_colours], axis=1)
                    stats['pointcloud'][f'{pcl_type}_depth_orig'] = wandb.Object3D(pcl_max_depth_combined)
                    for j, depth_j in enumerate(octree_local.pyramid_depths):
                        pcl_depth_j = get_octant_centroids_from_points(
                            octree_points_local, depth=depth_j, quantizer=self.params.model_params.quantizer
                        )
                        batch_mask_depth_j = octree_local.batch_id(depth_j, octree_local.nempty) == batch_idx
                        pcl_depth_j_masked = pcl_depth_j[batch_mask_depth_j].numpy()
                        # Colourise by last layer embeddings if available, else by height
                        if 'local_feats' in feats_and_attn_maps[-1]:
                            local_feats_depth_j = feats_and_attn_maps[block_idx]['local_feats'][depth_j]
                            local_feats_depth_j_masked = release_cuda(local_feats_depth_j[batch_mask_depth_j], to_numpy=True)
                            assert len(local_feats_depth_j_masked) == len(pcl_depth_j_masked)
                            pcl_depth_j_colours, local_feats_depth_j_variance = colourise_points_by_similarity(
                                local_feats_depth_j_masked, mode='pca', return_explained_variance=True
                            )
                            pcl_depth_j_colours *= 255.0
                        else:
                            pcl_depth_j_colours = colourise_points_by_height(pcl_depth_j_masked) * 255.0
                        pcl_depth_j_combined = np.concatenate([pcl_depth_j_masked, pcl_depth_j_colours], axis=1)
                        stats['pointcloud'][f'{pcl_type}_stage_{j}'] = wandb.Object3D(pcl_depth_j_combined)
                        stats['pca_variance'][f'{pcl_type}_stage_{j}'] = wandb.Histogram(local_feats_depth_j_variance)
            return stats
        def log_octformer() -> tp.Dict:
            """
            Log feats and attn maps for OctFormer variants
            """
            # Make absolutely sure that gradients aren't touched here
            with torch.no_grad():
                stats = {'ct_attn_map': {}, 'local_attn_map': {}, 'local_rpe': {},
                        'ct_token_unique_sim': {}, 'local_token_unique_sim': {},
                        'ct_token_sim_matrix': {}, 'local_token_sim_matrix': {},
                        'pointcloud': {}, 'pca_variance': {}}
                for j, depth_j in enumerate(feats_and_attn_maps.keys()): 
                    block_indices = np.unique(np.linspace(0, len(feats_and_attn_maps[depth_j])-1,
                                              VIZ_BLOCKS, dtype=np.int32)).tolist()
                    for block_idx in block_indices:
                        # TODO: implement CT logging for OctFormer
                        # if 'ct_attn' in ...
                        if 'local_attn' in feats_and_attn_maps[depth_j][block_idx]:
                            local_attn_i_depth_j = feats_and_attn_maps[depth_j][block_idx]['local_attn']
                            local_rpe_i_depth_j = None
                            if block_idx == 0:
                                stats['local_attn_map'][f'stage_{j}'] = {}
                                stats['local_rpe'][f'stage_{j}'] = {}
                            local_attn_maps_i_depth_j = local_attn_i_depth_j['attn_map']
                            # Also log RPE to compare differences
                            if 'rpe' in local_attn_i_depth_j:
                                local_rpe_i_depth_j = local_attn_i_depth_j['rpe']
                            num_heads = local_attn_maps_i_depth_j.size(1)
                            # Ensure window indices only index selected batch element, not all elements
                            # NOTE: NOT TESTED WELL FOR OCTFORMER WITH BATCH IDX > 0, SO USE WITH CAUTION
                            window_start_idx = 0 if BATCH_IDX == 0 else octree.batch_nnum_nempty[depth_j].cumsum(0)[BATCH_IDX-1].item()
                            window_end_idx = octree.batch_nnum_nempty[depth_j].cumsum(0)[BATCH_IDX].item()
                            window_indices = np.unique(np.linspace(window_start_idx, window_end_idx-1,
                                                       VIZ_LOCAL_WINDOWS, endpoint=False,  # last window may contain padding, so endpoint needs to be False
                                                       dtype=np.int32)).tolist() 
                            head_indices = np.unique(np.linspace(0, num_heads-1,
                                                     VIZ_HEADS, dtype=np.int32)).tolist()
                            for k, head_kdx in enumerate(head_indices):
                                for w, window_wdx in enumerate(window_indices):
                                    temp_attn_map = local_attn_maps_i_depth_j[window_wdx, head_kdx]
                                    if softmax:
                                        temp_attn_map = F.softmax(temp_attn_map, dim=-1)
                                    stats['local_attn_map'][f'stage_{j}'][f'block_{block_idx}_head_{k}_window_{w}'] \
                                        = wandb.Image(create_heatmap(temp_attn_map,
                                                                     title=f'Block {block_idx} - Head {k} - Window {w}'))
                                    plt.close()  # close fig to prevent going OOM
                                    if local_rpe_i_depth_j is not None:
                                        temp_rpe = local_rpe_i_depth_j[window_wdx, head_kdx]
                                        stats['local_rpe'][f'stage_{j}'][f'block_{block_idx}_head_{k}_window_{w}'] \
                                            = wandb.Image(create_heatmap(temp_rpe,
                                                                         title=f'Block {block_idx} - Head {k} - Window {w}')) 
                                        plt.close()
                        # Compute similarity metrics of tokens
                        if 'local_feats' in feats_and_attn_maps[depth_j][block_idx]:
                            token_batch_mask_depth_j = octree.batch_id(depth_j, octree.nempty) == BATCH_IDX
                            local_feats_i_depth_j = feats_and_attn_maps[depth_j][block_idx]['local_feats'].to(self.device)[token_batch_mask_depth_j]
                            if block_idx == 0:
                                stats['local_token_sim_matrix'][f'stage_{j}'] = {}
                                stats['local_token_unique_sim'][f'stage_{j}'] = {}
                            # Select only tokens from a single batch
                            temp_sim = rowwise_cosine_sim(local_feats_i_depth_j, local_feats_i_depth_j).cpu()
                            stats['local_token_sim_matrix'][f'stage_{j}'][f'block_{block_idx}'] = wandb.Image(create_heatmap(temp_sim,
                                                                                                                             title=f'Block {block_idx}'))
                            plt.close()
                            # Collect unique values of token similarity (off diagonal)
                            stats['local_token_unique_sim'][f'stage_{j}'][f'block_{block_idx}'] = wandb.Histogram(off_diagonal(temp_sim).numpy())

                # Log the point cloud itself
                octree_local = release_cuda(octree)
                octree_points_local = release_cuda(points)
                pcl_max_depth = get_octant_centroids_from_points(
                    octree_points_local, depth=self.params.octree_depth, quantizer=self.params.model_params.quantizer
                )
                # Log query and positive (queries and positives are neighbour pairs within the batch)
                assert BATCH_IDX % 2 == 0, "Queries are even indices, positives are odd indices"
                query_pos_indices = [BATCH_IDX, BATCH_IDX + 1]
                for i, batch_idx in enumerate(query_pos_indices):
                    pcl_type = 'query' if i == 0 else 'positive'
                    batch_mask_max_depth = octree_local.batch_id(self.params.octree_depth, octree_local.nempty) == batch_idx
                    pcl_max_depth_masked = pcl_max_depth[batch_mask_max_depth].numpy()
                    pcl_max_depth_colours = colourise_points_by_height(pcl_max_depth_masked) * 255.0
                    pcl_max_depth_combined = np.concatenate([pcl_max_depth_masked, pcl_max_depth_colours], axis=1)
                    stats['pointcloud'][f'{pcl_type}_depth_orig'] = wandb.Object3D(pcl_max_depth_combined)
                    for j, depth_j in enumerate(feats_and_attn_maps.keys()):
                        pcl_depth_j = get_octant_centroids_from_points(
                            octree_points_local, depth=depth_j, quantizer=self.params.model_params.quantizer
                        )
                        batch_mask_depth_j = octree_local.batch_id(depth_j, octree_local.nempty) == batch_idx
                        pcl_depth_j_masked = pcl_depth_j[batch_mask_depth_j].numpy()
                        # Colourise by last layer embeddings if available, else by height
                        if 'local_feats' in feats_and_attn_maps[depth_j][-1]:
                            local_feats_depth_j = feats_and_attn_maps[depth_j][block_idx]['local_feats']
                            local_feats_depth_j_masked = release_cuda(local_feats_depth_j[batch_mask_depth_j], to_numpy=True)
                            assert len(local_feats_depth_j_masked) == len(pcl_depth_j_masked)
                            pcl_depth_j_colours, local_feats_depth_j_variance = colourise_points_by_similarity(
                                local_feats_depth_j_masked, mode='pca', return_explained_variance=True
                            )
                            pcl_depth_j_colours *= 255.0
                        else:
                            pcl_depth_j_colours = colourise_points_by_height(pcl_depth_j_masked) * 255.0
                        pcl_depth_j_combined = np.concatenate([pcl_depth_j_masked, pcl_depth_j_colours], axis=1)
                        stats['pointcloud'][f'{pcl_type}_stage_{j}'] = wandb.Object3D(pcl_depth_j_combined)
                        stats['pca_variance'][f'{pcl_type}_stage_{j}'] = wandb.Histogram(local_feats_depth_j_variance)
            return stats

        if 'octformer' in self.params.model_params.model.lower():
            stats = log_octformer()
        elif 'hotformer' in self.params.model_params.model.lower():
            stats = log_hotformer()
        else:
            raise NotImplementedError
        self.logger.debug(f'Logged feats and attn maps in {time.time() - tic:.4f}s')
        return stats

    def global_training_step(self, global_batch, phase, loss_fn, qkv_loss_fn, num_embeddings_logged, mesa=0.0):
        assert phase in ['train', 'secondary_train', 'val']

        batch, positives_mask, negatives_mask = global_batch
        batch = to_device(batch, self.device, non_blocking=True, construct_octree_neigh=True)

        with torch.set_grad_enabled('train' in phase):
            y = self.model(batch, global_only=True)
            stats = self.model.stats.copy() if hasattr(self.model, 'stats') else {}

            embeddings = y['global']
            # Log stats related to feats and attn maps
            if 'train' in phase and 'feats_and_attn_maps' in y and self.params.wandb:
                feats_and_attn_maps = y.pop('feats_and_attn_maps')
                if 'octree' in y:
                    temp_stats = self.log_feats_and_attn_maps(feats_and_attn_maps, y['octree'], batch['points'])
                else:  # minkloc
                    temp_stats = self.log_feats(feats_and_attn_maps)
                del feats_and_attn_maps  # free memory from intermediate feats
                stats.update(temp_stats)

            loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
            temp_stats = self.tensors_to_numbers(temp_stats)
            stats.update(temp_stats)
            if 'train' in phase:
                # Compute MESA loss
                if mesa > 0.0:
                    with torch.no_grad():
                        ema_output = self.model_ema.module(batch, global_only=True)['global'].detach()
                    kd = kdloss(embeddings, ema_output)
                    mesa_loss = mesa * kd
                    loss += mesa_loss
                    stats['loss_mesa'] = mesa_loss.item()

                # Compute qkv std regularisation loss
                if qkv_loss_fn is not None:
                    qkv_loss, temp_stats = qkv_loss_fn(y, self.model)
                    temp_stats = self.tensors_to_numbers(temp_stats)
                    stats.update(temp_stats)
                    loss += qkv_loss

                loss.backward()

                # NOTE: Verify that EMA works correctly with metric loc model + moving update outside of global train step 
                # if self.model_ema is not None:
                #     self.model_ema.update(self.model)

            stats['loss_total'] = loss.item()

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        return stats, release_cuda(embeddings[:num_embeddings_logged])  # return first n embeddings for debugging


    def multistaged_global_training_step(self, global_batch, phase, loss_fn, qkv_loss_fn, num_embeddings_logged, mesa=0.0):
        # Training step using multistaged backpropagation algorithm as per:
        # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
        # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
        # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
        # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774
        if qkv_loss_fn is not None:
            raise NotImplementedError('QKV Losses not implemented for multistage backprop, set `batch_split_size` to 0')
        assert phase in ['train', 'secondary_train', 'val']

        batch, positives_mask, negatives_mask = global_batch

        # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
        # In training phase network is in the train mode to update BatchNorm stats
        embeddings_l = []
        embeddings_ema_l = []
        stats = {}
        with torch.set_grad_enabled(False):
            for i, minibatch in enumerate(batch):
                minibatch = to_device(minibatch, self.device, non_blocking=True, construct_octree_neigh=True)
                y = self.model(minibatch, global_only=True)
                embeddings_l.append(y['global'])
                if 'train' not in phase:
                    del minibatch, y
                    continue
                # Log stats related to feats and attn maps (only for first minibatch)
                if i == 0 and 'feats_and_attn_maps' in y and self.params.wandb:
                    feats_and_attn_maps = y.pop('feats_and_attn_maps')
                    if 'octree' in y:
                        temp_stats = self.log_feats_and_attn_maps(feats_and_attn_maps, y['octree'], minibatch['points'])
                    else:  # minkloc
                        temp_stats = self.log_feats(feats_and_attn_maps)
                    del feats_and_attn_maps  # free memory from intermediate feats
                    stats.update(temp_stats)
                # Compute MESA embeddings
                if mesa > 0.0:
                    ema_output = self.model_ema.module(minibatch, global_only=True)['global'].detach()
                    embeddings_ema_l.append(ema_output)
                del minibatch, y

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        # Stage 2 - compute gradient of the loss w.r.t embeddings
        embeddings = torch.cat(embeddings_l, dim=0)
        if 'train' in phase and mesa > 0.0:
            embeddings_ema = torch.cat(embeddings_ema_l, dim=0)

        with torch.set_grad_enabled('train' in phase):
            if 'train' in phase:
                embeddings.requires_grad_(True)
            loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
            temp_stats = self.tensors_to_numbers(temp_stats)
            stats.update(temp_stats)
            # Compute MESA loss
            if 'train' in phase and mesa > 0.0:
                kd = kdloss(embeddings, embeddings_ema)
                mesa_loss = mesa * kd
                loss += mesa_loss
                stats['loss_mesa'] = mesa_loss.item()
            # Compute qkv loss
            # NOTE: qkv loss is currently broken for multistage backprop. In
            #       stage 3, .backward() needs to be called w.r.t the gradient
            #       of all qkv_std values from all layers.
            # if qkv_loss_fn is not None:
            #     qkv_loss, temp_stats = qkv_loss_fn(y, self.model)
            #     temp_stats = self.tensors_to_numbers(temp_stats)
            #     stats.update(temp_stats)
            #     loss += qkv_loss
            stats['loss_total'] = loss.item()
            if 'train' in phase:
                loss.backward()
                embeddings_grad = embeddings.grad

        # Delete intermediary values
        embeddings_l, embeddings, embeddings_ema_l, embeddings_ema, y, loss = [None]*6

        # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
        # network parameters using cached gradient of the loss w.r.t embeddings
        if 'train' in phase:
            i = 0
            with torch.set_grad_enabled(True):
                for minibatch in batch:
                    minibatch = to_device(minibatch, self.device, non_blocking=True, construct_octree_neigh=True)
                    y = self.model(minibatch, global_only=True)
                    embeddings = y['global']
                    minibatch_size = len(embeddings)
                    # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                    # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                    # By default gradients are accumulated
                    embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                    i += minibatch_size
                    del minibatch, y

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        if embeddings is not None:
            return stats, release_cuda(embeddings[:num_embeddings_logged])  # return first n embeddings for debugging
        else:
            return stats, embeddings

    def local_training_step(self, local_batch, phase, local_loss_fn):
        assert phase in ['train', 'secondary_train', 'val']

        local_batch = to_device(local_batch, self.device, non_blocking=True, construct_octree_neigh=True)

        with torch.set_grad_enabled('train' in phase):
            output_dicts = self.model(local_batch)
            batch_local_loss = []
            batch_metrics = []
            # Compute loss for each batch item
            for ii, output_dict in enumerate(output_dicts):
                local_batch_ii = {'transform': local_batch['transform'][ii]}  # temp fix since loss func expects a single batch item
                temp_local_loss, local_metrics = local_loss_fn(output_dict, local_batch_ii)
                batch_local_loss.append(temp_local_loss)
                local_metrics = self.tensors_to_numbers(local_metrics)
                batch_metrics.append(local_metrics)

            # Average loss and metrics
            batch_local_loss = torch.stack(batch_local_loss).mean()
            batch_metrics = metrics_mean(batch_metrics)

        if 'train' in phase:
            batch_local_loss.backward()

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        return batch_metrics

    def do_train(self):        
        # set up dataloaders
        dataloaders = make_dataloaders(
            self.params,
            local=self.params.local.enable_local,
            validation=self.params.validation
        )

        global_loss_fn, qkv_loss_fn, local_loss_fn = make_losses(self.params)

        if self.params.batch_split_size is None or self.params.batch_split_size == 0:
            global_train_step_fn = self.global_training_step
        else:
            # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
            global_train_step_fn = self.multistaged_global_training_step

        ########################################################################
        # Initialize Weights&Biases logging service
        ########################################################################

        params_dict = {e: self.params.__dict__[e] for e in self.params.__dict__ if e != 'model_params'}
        model_params_dict = {"model_params." + e: self.params.model_params.__dict__[e] for e in self.params.model_params.__dict__}
        params_dict.update(model_params_dict)
        n_params = sum([param.nelement() for param in self.model.parameters()])
        params_dict['num_params'] = n_params
        if self.params.wandb and not self.params.debug:
            if WANDB_OFFLINE:
                self.trigger_sync = TriggerWandbSyncHook()  # callback to sync offline wandb dirs
            wandb.init(project='HOTFormerLoc', config=params_dict, id=self.wandb_id, resume="allow")
            self.wandb_id = wandb.run.id
            wandb.watch(self.model, log='all', log_freq=self.params.embeddings_log_freq)

        ########################################################################
        # Training Loop
        ########################################################################

        phases = ['train']
        if self.params.secondary_dataset_name is not None:
            phases.append('secondary_train')
        if self.params.validation:
            # Validation phase
            phases.append('val')

        # Training statistics
        stats = {e: [] for e in phases}

        for epoch in tqdm.tqdm(range(self.start_epoch, self.params.epochs + 1),
                               initial=self.start_epoch-1, total=self.params.epochs):
            metrics = {'train': {}, 'secondary_train': {}, 'val': {}, 'test': {}}  # Metrics for wandb reporting
            if epoch / self.params.epochs > self.params.mesa_start_ratio:
                mesa = self.params.mesa
            else:
                mesa = 0.0
            
            for phase in phases:
                self.logger.info(f"Begin {phase} phase")
                running_stats = []  # running stats for the current epoch and phase
                self.count_batches = 0
                epoch_embeddings = None
                epoch_stage_gradient_magnitudes = None

                if phase == 'train':
                    self.model.train()
                    global_phase = 'global_train'
                    local_phase = 'local_train'
                elif phase == 'secondary_train':
                    self.model.train()
                    global_phase = 'secondary_train'
                    local_phase = 'local_secondary_train'
                elif phase == 'val':
                    self.model.eval()
                    global_phase = 'global_val'
                    local_phase = 'local_val'
                else:
                    raise NotImplementedError()

                for global_batch, local_batch in zip(dataloaders[global_phase], dataloaders[local_phase]):
                    self.count_batches += 1
                    batch_stats = {}
                    if self.params.debug and self.count_batches > 2:
                        break
                    
                    self.logger.debug(f"Global batch {self.count_batches} start")
                    temp_global_stats, temp_global_embeddings = global_train_step_fn(
                        global_batch, phase, global_loss_fn, qkv_loss_fn,
                        self.params.num_embeddings_logged, mesa
                    )
                    self.logger.debug(f"Global batch {self.count_batches} end")
                    batch_stats['global'] = temp_global_stats
                    
                    if self.params.local.enable_local and local_batch is not None:
                        self.logger.debug(f"Local batch {self.count_batches} start")
                        temp_local_stats = self.local_training_step(
                            local_batch, phase, local_loss_fn,
                        )
                        batch_stats['local'] = temp_local_stats
                        self.logger.debug(f"Local batch {self.count_batches} end")

                    if 'train' in phase:
                        # Step optimizer after .backward called for global and local losses
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # Update EMA of model params for MESA loss
                        if self.model_ema is not None:
                            self.model_ema.update(self.model)

                    running_stats.append(batch_stats)
                    if (epoch % self.params.embeddings_log_freq == 0
                        and self.count_batches == 1 and 'train' in phase):  # log embeddings once per epoch
                        epoch_embeddings = temp_global_embeddings

                # Log average gradients per stage
                if 'train' in phase and not isinstance(self.model, (HOTFormerLoc, HOTFormerMetricLoc)):
                    # FIXME: currently broken for HOTFormerLoc
                    epoch_stage_gradient_magnitudes = self.log_stage_gradient_magnitudes(self.params.load_octree)

                # Compute mean stats for the phase
                epoch_stats = self.compute_mean_epoch_stats(running_stats)

                stats[phase].append(epoch_stats)
                self.print_stats(phase, epoch_stats)

                # Log metrics for wandb
                metrics[phase].update(self.get_wandb_metrics(epoch_stats))
                if epoch_stage_gradient_magnitudes is not None:
                    metrics[phase]['avg_stage_grad_mags'] = epoch_stage_gradient_magnitudes

            # ******* FINALIZE THE EPOCH *******
            self.curr_epoch += 1  # increment the epoch counter here to ensure next ckpt saves with correct epoch count
            if self.scheduler is not None:
                self.scheduler.step()
            
            if not self.params.debug:
                checkpoint_path = self.model_pathname + self.checkpoint_extension
                self.save_checkpoint(checkpoint_path)
                if self.params.save_freq > 0 and epoch % self.params.save_freq == 0:
                    epoch_pathname = f"{self.model_pathname}_e{epoch}.ckpt"
                    self.save_checkpoint(epoch_pathname)

            if self.params.eval_freq > 0 and epoch % self.params.eval_freq == 0:
                self.logger.debug("Begin evaluation")
                global_eval_stats, local_eval_stats = evaluate(
                    self.model,
                    self.device,
                    self.params,
                    log=False,
                    radius=self.params.eval_radius,
                    icp_refine=False,
                    local_max_eval_threshold=self.params.local.max_eval_threshold,
                    show_progress=self.params.verbose,
                    only_global=(not self.params.local.enable_local)
                )
                print_eval_stats(global_eval_stats, local_eval_stats)
                metrics['test'] = self.log_eval_stats(global_eval_stats, local_eval_stats)
                # store best AR@1 on all test sets
                radius_best = min(self.params.eval_radius)
                avg_AR_1 = metrics['test']['average']['recall@1'][radius_best]
                if avg_AR_1 > self.best_avg_AR_1:
                    self.logger.info(f"New best avg AR@1 at Epoch {epoch}: {self.best_avg_AR_1:.2f} -> {avg_AR_1:.2f}")
                    self.best_avg_AR_1 = avg_AR_1
                    if not self.params.debug:
                        best_model_pathname = f"{self.model_pathname}_best.ckpt"
                        self.save_checkpoint(best_model_pathname)

            if self.params.wandb and not self.params.debug:
                wandb.log(metrics)
                if WANDB_OFFLINE:
                    self.trigger_sync()

            if self.params.batch_expansion_th is not None:
                # Dynamic batch size expansion based on number of non-zero triplets
                # Ratio of non-zero triplets
                le_train_stats = stats['train'][-1]  # Last epoch training stats
                rnz = (
                    le_train_stats['global']['num_non_zero_triplets']
                    / le_train_stats['global']['num_triplets']
                )
                if rnz < self.params.batch_expansion_th:
                    dataloaders['global_train'].batch_sampler.expand_batch()
                if self.params.secondary_dataset_name is not None:
                    le_secondary_train_stats = stats['secondary_train'][-1]
                    rnz = (
                        le_secondary_train_stats['global']['num_non_zero_triplets']
                        / le_secondary_train_stats['global']['num_triplets']
                    )
                    if rnz < self.params.batch_expansion_th:
                        dataloaders['secondary_train'].batch_sampler.expand_batch()

        # Save final model weights
        if not self.params.debug:
            final_model_path = self.model_pathname + '_final.ckpt'
            self.save_checkpoint(final_model_path)

        # Evaluate the final
        final_global_stats, final_local_stats = evaluate(
            self.model,
            self.device,
            self.params,
            log=False,
            radius=self.params.eval_radius,
            icp_refine=False,
            local_max_eval_threshold=self.params.local.max_eval_threshold,
            show_progress=self.params.verbose,
            only_global=(not self.params.local.enable_local)
        )
        print_eval_stats(final_global_stats, final_local_stats)

        # Append key experimental metrics to experiment summary file
        if not self.params.debug:
            model_params_name = os.path.split(self.params.model_params.model_params_path)[1]
            config_name = os.path.split(self.params.params_path)[1]
            model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
            prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

            write_eval_stats(f"metloc_sgv_{self.params.dataset_name}_split_results.txt", prefix, final_global_stats, final_local_stats)        

            if self.params.wandb and WANDB_OFFLINE:
                self.trigger_sync()

        # Return optimization value (to minimize)
        return (1 - self.best_avg_AR_1/100.0)

    @staticmethod
    def compute_mean_epoch_stats(running_stats):
        epoch_stats = {}
        for substep in running_stats[0]:
            epoch_stats[substep] = {}
            for key in running_stats[0][substep]:
                try:
                    temp = [e[substep][key] for e in running_stats]
                except KeyError:  # special case when value is only logged once during training
                    epoch_stats[substep][key] = running_stats[0][substep][key]
                else:
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        # Mean value per vector element
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)
        return epoch_stats

    def get_wandb_metrics(self, epoch_stats: tp.Dict[str, tp.Dict]) -> tp.Dict:
        metrics = {}
        # Global Metrics
        metrics['loss1'] = epoch_stats['global']['loss']
        metrics['loss_total'] = epoch_stats['global']['loss_total']
        if 'num_non_zero_triplets' in epoch_stats['global']:
            metrics['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']
        if 'positive_ranking' in epoch_stats['global']:
            metrics['positive_ranking'] = epoch_stats['global']['positive_ranking']
        if 'recall' in epoch_stats['global']:
            metrics['recall@1'] = epoch_stats['global']['recall'][1]
        if 'ap' in epoch_stats['global']:
            metrics['AP'] = epoch_stats['global']['ap']
        if 'loss_mesa' in epoch_stats['global']:
            metrics['loss_mesa'] = epoch_stats['global']['loss_mesa']
        if 'local_qkv_std_loss' in epoch_stats['global']:
            metrics['local_qkv_std_loss'] = epoch_stats['global']['local_qkv_std_loss']
        if 'local_qkv_std' in epoch_stats['global']:
            metrics['local_qkv_std'] = epoch_stats['global']['local_qkv_std']
        if 'rt_qkv_std_loss' in epoch_stats['global']:
            metrics['rt_qkv_std_loss'] = epoch_stats['global']['rt_qkv_std_loss']
        if 'rt_qkv_std' in epoch_stats['global']:
            metrics['rt_qkv_std'] = epoch_stats['global']['rt_qkv_std']
        if 'qkv_weight_norm_loss' in epoch_stats['global']:
            metrics['qkv_weight_norm_loss'] = epoch_stats['global']['qkv_weight_norm_loss']
        if 'qkv_weight_norm' in epoch_stats['global']:
            metrics['qkv_weight_norm'] = epoch_stats['global']['qkv_weight_norm']
        if 'rt_attn_map' in epoch_stats['global']:
            metrics['rt_attn_map'] = epoch_stats['global']['rt_attn_map']
        if 'local_attn_map' in epoch_stats['global']:
            metrics['local_attn_map'] = epoch_stats['global']['local_attn_map']
        if 'local_rpe' in epoch_stats['global']:
            metrics['local_rpe'] = epoch_stats['global']['local_rpe']
        if 'rt_token_unique_sim' in epoch_stats['global']:
            metrics['rt_token_unique_sim'] = epoch_stats['global']['rt_token_unique_sim']
        if 'local_token_unique_sim' in epoch_stats['global']:
            metrics['local_token_unique_sim'] = epoch_stats['global']['local_token_unique_sim']
        if 'rt_token_sim_matrix' in epoch_stats['global']:
            metrics['rt_token_sim_matrix'] = epoch_stats['global']['rt_token_sim_matrix']
        if 'local_token_sim_matrix' in epoch_stats['global']:
            metrics['local_token_sim_matrix'] = epoch_stats['global']['local_token_sim_matrix']
        if 'pointcloud' in epoch_stats['global']:
            metrics['pointcloud'] = epoch_stats['global']['pointcloud']
        if 'pca_variance' in epoch_stats['global']:
            metrics['pca_variance'] = epoch_stats['global']['pca_variance']
        # Local Metrics
        if not (self.params.local.enable_local and 'local' in epoch_stats):
            return metrics
        metrics['local'] = {}
        if 'loss' in epoch_stats['local']:
            metrics['local']['loss'] = epoch_stats['local']['loss']
        if 'coarse_loss' in epoch_stats['local']:
            metrics['local']['coarse_loss'] = epoch_stats['local']['coarse_loss']
        if 'fine_loss' in epoch_stats['local']:
            metrics['local']['fine_loss'] = epoch_stats['local']['fine_loss']
        if 'PIR' in epoch_stats['local']:
            metrics['local']['coarse_IR'] = epoch_stats['local']['PIR']
        if 'IR' in epoch_stats['local']:
            metrics['local']['fine_IR'] = epoch_stats['local']['IR']
        if 'RRE' in epoch_stats['local']:
            metrics['local']['RRE'] = epoch_stats['local']['RRE']
        if 'RTE' in epoch_stats['local']:
            metrics['local']['RTE'] = epoch_stats['local']['RTE']
        if 'RR' in epoch_stats['local']:
            metrics['local']['RR'] = epoch_stats['local']['RR']
        return metrics
# Warsaw University of Technology
# Train MinkLoc model

import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pathlib
import typing as tp
import wandb
import submitit
import matplotlib.pyplot as plt
import seaborn as sns
from timm.utils import ModelEmaV3
from timm.optim.lamb import Lamb
# import wandb_osh
# from wandb_osh.hooks import TriggerWandbSyncHook
os.environ["WANDB__SERVICE_WAIT"] = "300"  # prevent crash if wandb is slow
# wandb_osh.set_log_level("ERROR")

from misc.utils import TrainingParams, get_datetime, set_seed, update_params_from_dict
from models.losses.loss import make_losses, kdloss
from models.model_factory import model_factory
from models.hotformerloc import HOTFormerLoc
from models.octree import OctreeT
from dataset.dataset_utils import make_dataloaders
from eval.pnv_evaluate import evaluate, print_eval_stats, pnv_write_eval_stats
from eval.vis_utils import remove_rt_attn_padding, rowwise_cosine_sim, off_diagonal, \
    get_octree_points_and_windows, colourise_points_by_height, colourise_points_by_similarity

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
        checkpoint_path = kwargs.get('checkpoint_path')
        self.resume = checkpoint_path is not None        
        # Set params for hyperparam search
        self.params = params
        if not self.resume:
            if self.params.hyperparam_search:
                if len(args) == 1 and isinstance(args[0], dict):  # This is required for submitit job arrays currently
                    kwargs = args[0]
                assert kwargs != {}, 'No valid hyperparams were provided for search'
                self.params = update_params_from_dict(self.params, kwargs)
        self.params.print()
        
        # Seed RNG
        set_seed()
        print('Determinism: Enabled')

        # Initialise model, optimiser, and scheduler 
        self.init_model_optim_sched()
        
        # Load state from ckpt if resubmitted, otherwise start from scratch
        if self.resume:
            self.model_pathname = checkpoint_path.split(self.checkpoint_extension)[0]
            state = torch.load(checkpoint_path)
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
                print("Invalid checkpoint file provided. Only files ending in '.ckpt' are valid for resuming training.")
            print(f'Resuming training of {self.model_pathname} from epoch {self.start_epoch}')
        else:
            # Create model class
            s = get_datetime()
            model_name = self.params.model_params.model + '_' + s
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
                    
        # Begin train loop
        self.do_train()

    def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> submitit.helpers.DelayedSubmission:
        """
        This function is called asynchronously by submitit when a job is timed
        out. Dumps the current model state to disk and returns a
        DelayedSubmission to resubmit the job.
        """
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
        print(f"[INFO] Saving checkpoint to {checkpoint_path}", flush=True)
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

    def init_model_optim_sched(self):
        """
        Initialise the model, optimiser, and scheduler from the provided parameters.
        """
        self.model = model_factory(self.params.model_params)

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
            self.model_ema = ModelEmaV3(self.model, decay=0.9998)

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
        s = f"{phase}  total loss: {stats['loss_total']:.4f}   PR loss: {stats['loss']:.4f}   "
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

        print(s, flush=True)

    def print_stats(self, phase, stats):
        self.print_global_stats(phase, stats['global'])
    
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

    def log_eval_stats(self, stats):
        eval_stats = {}
        for database_name in stats:
            eval_stats[database_name] = {}
            eval_stats[database_name]['recall@1%'] = stats[database_name]['ave_one_percent_recall']
            eval_stats[database_name]['recall@1'] = stats[database_name]['ave_recall'][0]
            eval_stats[database_name]['MRR'] = stats[database_name]['ave_mrr']
        return eval_stats

    def log_feats_and_attn_maps(self, feats_and_attn_maps: tp.List, octree: OctreeT,
                                softmax=False) -> tp.Dict:
        """
        Logs various things including attention maps, average token similarity.
        """
        VIZ_BLOCKS = 5
        VIZ_HEADS = 2
        VIZ_LOCAL_WINDOWS = 1
        BATCH_IDX = 0  # just take samples from the first batch element for attn maps
        # Only log once per epoch
        if self.count_batches > 1:
            return {}
        
        def create_heatmap(attn_map: torch.Tensor, ticklabels: tp.Optional[tp.List] = None,
                           title: tp.Optional[str] = None) -> plt.Figure:
            if ticklabels is None:
                ticklabels = 'auto'
            eps = 100  # epsilon for filtering masked tokens (as masked tokens may have a slightly different value after attention)
            CMAP = 'viridis'
            fig = plt.figure(figsize=(6,5))
            # Clip masked values to prevent them overpowering the attn map
            vmin = attn_map[attn_map > (octree.invalid_mask_value + eps)].min().item()
            ax = sns.heatmap(attn_map, cmap=CMAP, vmin=vmin, xticklabels=ticklabels, yticklabels=ticklabels)
            if title is not None:
                ax.set_title(title)
            return fig
        def get_rt_heatmap_ticklabels() -> tp.List:
            """
            Create ticklabels for relay token heatmaps to indicate where each
            octree level starts/ends.
            """
            num_rt = octree.batch_num_relay_tokens_combined[BATCH_IDX]
            rt_idx_per_depth_list = []
            for depth_j in octree.pyramid_depths:
                rt_idx_per_depth_list.append(octree.batch_num_windows[depth_j][BATCH_IDX].item())
            ticklabels = ['' for _ in range(num_rt)]
            num_rt_depth_cumsum = np.cumsum(rt_idx_per_depth_list)
            for num_rt_depth_j in num_rt_depth_cumsum[:-1]:
                ticklabels[num_rt_depth_j] = num_rt_depth_j
            return ticklabels

        # Make absolutely sure that gradients aren't touched here
        with torch.no_grad():
            stats = {'rt_attn_map': {}, 'local_attn_map': {}, 'local_rpe': {},
                     'rt_token_unique_sim': {}, 'local_token_unique_sim': {},
                     'rt_token_sim_matrix': {}, 'local_token_sim_matrix': {},
                     'pointcloud': {}}
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
                            = wandb.Image(create_heatmap(temp_attn_map.cpu(), rt_ticklabels,
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
            pcl_max_depth, _, _ = get_octree_points_and_windows(
                octree, depth=self.params.octree_depth, params=self.params
            )
            batch_mask_max_depth = octree.batch_id(self.params.octree_depth, octree.nempty) == BATCH_IDX
            pcl_max_depth_masked = pcl_max_depth[batch_mask_max_depth].cpu().numpy()
            pcl_max_depth_colours = colourise_points_by_height(pcl_max_depth_masked) * 255.0
            pcl_max_depth_combined = np.concatenate([pcl_max_depth_masked, pcl_max_depth_colours], axis=1)
            stats['pointcloud']['depth_orig'] = wandb.Object3D(pcl_max_depth_combined)
            for j, depth_j in enumerate(octree.pyramid_depths):
                pcl_depth_j, _, _ = get_octree_points_and_windows(
                    octree, depth=depth_j, params=self.params
                )
                batch_mask_depth_j = octree.batch_id(depth_j, octree.nempty) == BATCH_IDX
                pcl_depth_j_masked = pcl_depth_j[batch_mask_depth_j].cpu().numpy()
                # Colourise by last layer embeddings if available, else by height
                if 'local_feats' in feats_and_attn_maps[-1]:
                    local_feats_depth_j = feats_and_attn_maps[block_idx]['local_feats'][depth_j]
                    local_feats_depth_j_masked = local_feats_depth_j[batch_mask_depth_j.cpu()].numpy()
                    assert len(local_feats_depth_j_masked) == len(pcl_depth_j_masked)
                    pcl_depth_j_colours = colourise_points_by_similarity(local_feats_depth_j_masked, mode='pca') * 255.0
                else:
                    pcl_depth_j_colours = colourise_points_by_height(pcl_depth_j_masked) * 255.0
                pcl_depth_j_combined = np.concatenate([pcl_depth_j_masked, pcl_depth_j_colours], axis=1)
                stats['pointcloud'][f'stage_{j}'] = wandb.Object3D(pcl_depth_j_combined)
                            
        return stats

    def training_step(self, global_iter, phase, loss_fn, qkv_loss_fn, num_embeddings_logged, mesa=0.0):
        assert phase in ['train', 'val']

        batch, positives_mask, negatives_mask = next(global_iter)

        if self.params.verbose:
            print("[INFO] Batch loaded, begin forward pass", flush=True)

        batch = {e: batch[e].to(self.device, non_blocking=True) for e in batch}

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            y = self.model(batch)
            stats = self.model.stats.copy() if hasattr(self.model, 'stats') else {}

            embeddings = y['global']
            # Log stats related to feats and attn maps
            if 'feats_and_attn_maps' in y:
                temp_stats = self.log_feats_and_attn_maps(y['feats_and_attn_maps'], y['octree'])
                stats.update(temp_stats)

            loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
            temp_stats = self.tensors_to_numbers(temp_stats)
            stats.update(temp_stats)
            if phase == 'train':
                # Compute MESA loss
                if mesa > 0.0:
                    with torch.no_grad():
                        ema_output = self.model_ema.module(batch)['global'].detach()
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

                stats['loss_total'] = loss.item()
                loss.backward()
                self.optimizer.step()
                
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        return stats, embeddings[:num_embeddings_logged].detach().cpu()  # return first n embeddings for debugging


    def multistaged_training_step(self, global_iter, phase, loss_fn, qkv_loss_fn, num_embeddings_logged, mesa=0.0):
        # Training step using multistaged backpropagation algorithm as per:
        # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
        # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
        # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
        # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774
        if qkv_loss_fn is not None:
            raise NotImplementedError('QKV Losses not implemented for multistage backprop, set `batch_split_size` to 0')
        assert phase in ['train', 'val']
        batch, positives_mask, negatives_mask = next(global_iter)
        
        if self.params.verbose:
            print("[INFO] Batch loaded, begin forward pass", flush=True)

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
        # In training phase network is in the train mode to update BatchNorm stats
        embeddings_l = []
        embeddings_ema_l = []
        stats = {}
        with torch.set_grad_enabled(False):
            for i, minibatch in enumerate(batch):
                minibatch = {e: minibatch[e].to(self.device, non_blocking=True) for e in minibatch}
                y = self.model(minibatch)
                embeddings_l.append(y['global'])
                # Log stats related to feats and attn maps (only for first minibatch)
                if i == 0 and 'feats_and_attn_maps' in y:
                    temp_stats = self.log_feats_and_attn_maps(y['feats_and_attn_maps'], y['octree'])
                    stats.update(temp_stats)
                # Compute MESA embeddings
                if mesa > 0.0:
                    ema_output = self.model_ema.module(minibatch)['global'].detach()
                    embeddings_ema_l.append(ema_output)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        # Stage 2 - compute gradient of the loss w.r.t embeddings
        embeddings = torch.cat(embeddings_l, dim=0)
        if mesa > 0.0:
            embeddings_ema = torch.cat(embeddings_ema_l, dim=0)

        with torch.set_grad_enabled(phase == 'train'):
            if phase == 'train':
                embeddings.requires_grad_(True)
            loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
            temp_stats = self.tensors_to_numbers(temp_stats)
            stats.update(temp_stats)
            # Compute MESA loss
            if mesa > 0.0:
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
            if phase == 'train':
                loss.backward()
                embeddings_grad = embeddings.grad

        # Delete intermediary values
        embeddings_l, embeddings, embeddings_ema_l, embeddings_ema, y, loss = [None]*6

        # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
        # network parameters using cached gradient of the loss w.r.t embeddings
        if phase == 'train':
            self.optimizer.zero_grad()
            i = 0
            with torch.set_grad_enabled(True):
                for minibatch in batch:
                    minibatch = {e: minibatch[e].to(self.device, non_blocking=True) for e in minibatch}
                    y = self.model(minibatch)
                    embeddings = y['global']
                    minibatch_size = len(embeddings)
                    # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                    # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                    # By default gradients are accumulated
                    embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                    i += minibatch_size

                self.optimizer.step()
                
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        if embeddings is not None:
            return stats, embeddings[:num_embeddings_logged].detach().cpu()  # return first n embeddings for debugging
        else:
            return stats, embeddings

    def do_train(self):        
        # set up dataloaders
        dataloaders = make_dataloaders(self.params, validation=self.params.validation)

        loss_fn, qkv_loss_fn = make_losses(self.params)

        if self.params.batch_split_size is None or self.params.batch_split_size == 0:
            train_step_fn = self.training_step
        else:
            # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
            train_step_fn = self.multistaged_training_step

        ###########################################################################
        # Initialize Weights&Biases logging service
        ###########################################################################

        params_dict = {e: self.params.__dict__[e] for e in self.params.__dict__ if e != 'model_params'}
        model_params_dict = {"model_params." + e: self.params.model_params.__dict__[e] for e in self.params.model_params.__dict__}
        params_dict.update(model_params_dict)
        n_params = sum([param.nelement() for param in self.model.parameters()])
        params_dict['num_params'] = n_params
        if not self.params.debug:
            # trigger_sync = TriggerWandbSyncHook()  # callback to sync offline wandb dirs
            wandb.init(project='HOTFormerLoc', config=params_dict, id=self.wandb_id, resume="allow")
            self.wandb_id = wandb.run.id
            wandb.watch(self.model, log='all', log_freq=self.params.embeddings_log_freq)

        ###########################################################################
        #
        ###########################################################################

        # Training statistics
        stats = {'train': [], 'eval': []}

        if 'val' in dataloaders:
            # Validation phase
            phases = ['train', 'val']
            stats['val'] = []
        else:
            phases = ['train']

        for epoch in tqdm.tqdm(range(self.start_epoch, self.params.epochs + 1),
                               initial=self.start_epoch-1, total=self.params.epochs):
            metrics = {'train': {}, 'val': {}, 'test': {}}      # Metrics for wandb reporting
            if epoch / self.params.epochs > self.params.mesa_start_ratio:
                mesa = self.params.mesa
            else:
                mesa = 0.0
            
            for phase in phases:
                if self.params.verbose:
                    print(f"[INFO] Begin {phase} phase", flush=True)
                running_stats = []  # running stats for the current epoch and phase
                self.count_batches = 0
                epoch_embeddings = None
                epoch_stage_gradient_magnitudes = None

                if phase == 'train':
                    global_iter = iter(dataloaders['train'])
                else:
                    global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

                # TODO: Need to load a specific submap (or set of them) for 
                #       logging attn maps and token similarity. Can use existing
                #       dataloaders with a custom dataset/batch sampler.

                while True:
                    self.count_batches += 1
                    batch_stats = {}
                    if self.params.debug and self.count_batches > 2:
                        break
                    
                    if self.params.verbose:
                        print(f"[INFO] Processing {phase} batch: {self.count_batches}",
                            flush=True)

                    try:
                        temp_stats, temp_embeddings = train_step_fn(
                            global_iter, phase, loss_fn, qkv_loss_fn,
                            self.params.num_embeddings_logged, mesa
                        )
                        batch_stats['global'] = temp_stats

                    except StopIteration:
                        # Terminate the epoch when one of dataloders is exhausted
                        break

                    running_stats.append(batch_stats)
                    if (epoch % self.params.embeddings_log_freq == 0
                        and self.count_batches == 1 and phase == 'train'):  # log embeddings once per epoch
                        epoch_embeddings = temp_embeddings

                # Log average gradients per stage
                if phase == 'train' and not isinstance(self.model, HOTFormerLoc):
                    # FIXME: currently broken for HOTFormerLoc
                    epoch_stage_gradient_magnitudes = self.log_stage_gradient_magnitudes(self.params.load_octree)

                # Compute mean stats for the phase
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

                stats[phase].append(epoch_stats)
                self.print_stats(phase, epoch_stats)

                # Log metrics for wandb
                metrics[phase].update(self.get_wandb_metrics(epoch_stats))
                if epoch_stage_gradient_magnitudes is not None:
                    metrics[phase]['avg_stage_grad_mags'] = epoch_stage_gradient_magnitudes

                # TODO: currently broken, need to debug why wandb isn't logging correctly
                # if epoch_embeddings is not None:
                #     embeddings_dataframe = pd.DataFrame(epoch_embeddings, columns=[f'D{i}' for i in range(256)])
                #     metrics['embeddings'] = wandb.Table(dataframe=embeddings_dataframe)

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
                if self.params.verbose:
                    print("[INFO] Begin evaluation", flush=True)
                eval_stats = evaluate(self.model, self.device, self.params,
                                      log=False, show_progress=self.params.verbose)
                print_eval_stats(eval_stats)
                metrics['test'] = self.log_eval_stats(eval_stats)
                # store best AR@1 on all test sets
                avg_AR_1 = metrics['test']['average']['recall@1']
                if avg_AR_1 > self.best_avg_AR_1:
                    print(f"New best avg AR@1 at Epoch {epoch}: {self.best_avg_AR_1:.2f} -> {avg_AR_1:.2f}")
                    self.best_avg_AR_1 = avg_AR_1
                    if not self.params.debug:
                        best_model_pathname = f"{self.model_pathname}_best.ckpt"
                        self.save_checkpoint(best_model_pathname)

            if not self.params.debug:
                wandb.log(metrics)
                # trigger_sync()

            if self.params.batch_expansion_th is not None:
                # Dynamic batch size expansion based on number of non-zero triplets
                # Ratio of non-zero triplets
                le_train_stats = stats['train'][-1]  # Last epoch training stats
                rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
                if rnz < self.params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

        print('')

        # Save final model weights
        if not self.params.debug:
            final_model_path = self.model_pathname + '_final.ckpt'
            self.save_checkpoint(final_model_path)

        # Evaluate the final
        # PointNetVLAD datasets evaluation protocol
        stats = evaluate(self.model, self.device, self.params, log=False,
                         show_progress=self.params.verbose)
        print_eval_stats(stats)

        print('.')

        # Append key experimental metrics to experiment summary file
        if not self.params.debug:
            model_params_name = os.path.split(self.params.model_params.model_params_path)[1]
            config_name = os.path.split(self.params.params_path)[1]
            model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
            prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

            pnv_write_eval_stats(f"pnv_{self.params.dataset_name}_results.txt", prefix, stats)        
            # trigger_sync()

        # Return optimization value (to minimize)
        return (1 - self.best_avg_AR_1/100.0)

    def get_wandb_metrics(self, epoch_stats):
        metrics = {}
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
        return metrics

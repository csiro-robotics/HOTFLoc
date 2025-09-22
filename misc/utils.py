# Warsaw University of Technology
#
# Adapted for HOTFormerLoc by Ethan Griffiths

import os
import configparser
import time
import pickle

from easydict import EasyDict as edict

from dataset.quantization import PolarQuantizer, CartesianQuantizer
from dataset.coordinate_utils import CylindricalCoordinates


class ModelParams:
    def __init__(self, model_params_path, local=False):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = params.get('coordinates', 'polar')
        assert self.coordinates in ['polar', 'cartesian', 'cylindrical'], f'Unsupported coordinates: {self.coordinates}'

        if any(model in self.model.lower() for model in ('octformer', 'hotformer')):
            if 'cartesian' in self.coordinates:
                self.quantizer = None
            elif 'cylindrical' in self.coordinates:
                self.quantizer = CylindricalCoordinates(use_octree=True)
            else:
                raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")
        else:
            if 'polar' in self.coordinates:
                # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
                self.quantization_step = tuple([float(e) for e in params['quantization_step'].split(',')])
                assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
                self.quantizer = PolarQuantizer(quant_step=self.quantization_step)
            elif 'cartesian' in self.coordinates:
                # Single quantization step for cartesian coordinates
                self.quantization_step = params.getfloat('quantization_step')
                self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
            else:
                raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")

        # Use cosine similarity instead of Euclidean distance
        # When Euclidean distance is used, embedding normalization is optional
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)
        self.feature_size = params.getint('feature_size', 256)
        self.pooling = params.get('pooling', 'GeM')
        self.k_pooled_tokens = params.get('k_pooled_tokens', '64')  # number of tokens to pool to when using attentional pooling
        if self.k_pooled_tokens.isdigit():
            self.k_pooled_tokens = int(self.k_pooled_tokens)
        else:
            self.k_pooled_tokens = tuple([int(e) for e in params['k_pooled_tokens'].split(',')])
        self.num_top_down = params.getint('num_top_down', 1)
        self.return_feats_and_attn_maps = params.getboolean('return_feats_and_attn_maps', True)  # outputs feats and attn maps from each block of HOTFormerLoc (or MinkLoc)
        self.strict_loading = params.getboolean('strict_loading', True)  # Enable strict loading of weights (keys must match current model exactly)
        self.scale_grads = params.getboolean('scale_grads', False)  # Enables gradient scaling to prevent gradient underflow
        self.freeze_hotformerloc = params.getboolean('freeze_hotformerloc', False)  # Freeze HOTFloc layers, only train metric loc

        # Metric loc params
        if local:
            self.coarse_idx = tuple([int(e) for e in params['coarse_idx'].split(',')])
            self.fine_idx = params.getint('fine_idx')
            assert self.coarse_idx is not None and self.fine_idx is not None
            if 'coarse_feat_embed_dim' in params:  # Dimension to project coarse features with MLP before metric localisation and/or re-ranking (None to disable)
                self.coarse_feat_embed_dim = tuple([int(e) for e in params['coarse_feat_embed_dim'].split(',')])
            else:
                self.coarse_feat_embed_dim = None
            if self.coarse_feat_embed_dim is not None:
                assert len(self.coarse_feat_embed_dim) == len(self.coarse_idx), 'Expected same length for coarse idx and embed dim'
            self.fine_feat_embed_dim = params.getint('fine_feat_embed_dim', None)  # Dimension to project fine features with MLP before metric localisation (None to disable)
            self.metloc_mlp_ratio = params.getfloat('metloc_mlp_ratio', 2.0)  # Hidden dim ratio of MLP used for coarse and/or fine features

        if 'minkloc' in self.model.lower():
        #######################################################################
        # MinkLoc params
        #######################################################################
            # Size of the local features from backbone network (only for MinkNet based models)
            if 'planes' in params:
                self.planes = tuple([int(e) for e in params['planes'].split(',')])
            else:
                self.planes = tuple([32, 64, 64])

            if 'layers' in params:
                self.layers = tuple([int(e) for e in params['layers'].split(',')])
            else:
                self.layers = tuple([1, 1, 1])

            self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)
            self.block = params.get('block', 'BasicBlock')

        elif any(model in self.model.lower() for model in ('octformer', 'hotformer')):
            #######################################################################
            # OctFormer params
            #######################################################################
            if 'channels' in params:  # num channels per OctFormer stage
                self.channels = tuple([int(e) for e in params['channels'].split(',')])
            else:
                self.channels = tuple([96, 192, 384, 384])
            if 'num_blocks' in params:  # num OctFormer blocks per stage
                self.num_blocks = tuple([int(e) for e in params['num_blocks'].split(',')])
            else:
                self.num_blocks = tuple([2, 2, 6, 2])  # default to OctFormer-small
            if 'num_heads' in params:  # num attention heads per stage
                self.num_heads = tuple([int(e) for e in params['num_heads'].split(',')])
            else:
                self.num_heads = None
            self.patch_size = params.getint('patch_size', 32)  # size of window attention patch
            self.dilation = params.getint('dilation', 4)  # dilation value for octree attention
            self.ct_size = params.getint('ct_size', 1)  # carrier token size, if using HAT layers
            self.ct_propagation = params.getboolean('ct_propagation', False)  # propagate ct features to local features at end of stage
            self.ct_propagation_scale = params.getfloat('ct_propagation_scale', None)  # learnable scalar multiplier for ct propagation step
            self.ct_rpe_init = params.getboolean('ct_rpe_init', False)  # use learnable value for CTs in RPE (instead of zero)
            self.rt_init_type = params.get('rt_init_type', 'avg_pool')  # type of initialisation to use for relay tokens
            assert self.rt_init_type.lower() in ['avg_pool', 'max_pool', 'learnable']
            self.rt_class_token = params.getboolean('rt_class_token', False)  # use a class token in RTSA
            self.ADaPE_mode = params.get('ADaPE_mode', None)  # Use Absolute Distribution-aware Position Encoding (ADaPE) during carrier token attention. Mode (valid values: ['pos','var','cov']) determines whether position, variance, or covariance is used (cumulative aggregation of those three)
            self.ADaPE_use_accurate_point_stats = params.getboolean('ADaPE_use_accurate_point_stats', False)  # Use accurate point statistics when computing ADaPE (by default just takes octant centroids instead of considering true point distribution)
            self.drop_path = params.getfloat('drop_path', 0.5)  # stochastic depth dropout
            self.input_features = params.get('input_features', 'P')  # P for global position, D for local displacement (check docs)
            self.downsample_input_embeddings = params.getboolean('downsample_input_embeddings', True)
            self.num_input_downsamples = params.getint('num_input_downsamples', 2)  # number of downsampling stages in ConvEmbed
            self.disable_RPE = params.getboolean('disable_RPE', False)
            self.conv_norm = params.get('conv_norm', 'batchnorm')  # choose normalisation layer after convolution layers
            assert self.conv_norm in ['batchnorm', 'layernorm', 'powernorm']
            self.layer_scale = params.getfloat('layer_scale', None)  # coefficient to initialise learnable channel-wise scalar multipliers for attention outputs, or None to disable this.
            self.grad_checkpoint = params.getboolean('grad_checkpoint', True)
            if 'qkv_init' in params:
                self.qkv_init = list([e for e in params['qkv_init'].split(',')])  # method of initialisation to use for qkv linear layers
                if len(self.qkv_init) > 1:
                    self.qkv_init[1] = None if self.qkv_init[1] == 'None' else float(self.qkv_init[1])
            else:
                self.qkv_init = ['trunc_normal', 0.02]  # Second value is std dev, but is optional and can be different depening on initialisation parameters
            self.xcpe = params.getboolean('xCPE', False)  # Use xCPE instead of CPE (from PointTransformerV3)

            if any(model in self.model.lower() for model in ('hotformerloc', 'hotformermetricloc')):
                #######################################################################
                # HOTFormerLoc-specific params
                #######################################################################
                self.num_pyramid_levels = params.getint('num_pyramid_levels', 3)  # number of octree levels to consider for hierarchical attention.
                self.num_octf_levels = params.getint('num_octf_levels', 1)  # number of octformer levels to process local features before hierarchical attention
                self.disable_rt = params.getboolean('disable_rt', False)  # Disable all relay token components, and process HOTFormerLoc with solely local attention (with dilation re-enabled).
                # Re-ranking
                self.rerank_mode = params.get('rerank_mode', None)  # Type of re-ranking to do
                if self.rerank_mode is not None:
                    self.rerank_mode = self.rerank_mode.lower()
                    if self.rerank_mode not in ('relay_token_gc', 'relay_token_local_gc', 'local_hierarchical_gc', 'sgv'):
                        raise ValueError('Invalid re-ranking mode')
                if 'rerank_indices' in params:  # Indices (relative to feature pyramid) of relay token stages to use for re-ranking. Negative indices allowed.
                    self.rerank_indices = tuple([int(e) for e in params['rerank_indices'].split(',')])
                else:
                    self.rerank_indices = (0,)
                if 'rerank_feat_embed_dim' in params:  # Dimension to project coarse features with MLP before metric localisation and/or re-ranking (None to disable) 
                    self.rerank_feat_embed_dim = tuple([int(e) for e in params['rerank_feat_embed_dim'].split(',')])
                else:
                    self.rerank_feat_embed_dim = None
                if 'rerank_rt_attn_topk' in params:  # Top-k attn vals to keep from each pyramid level of relay tokens (must be same length as RT indices)
                    self.rerank_rt_attn_topk = tuple([int(e) for e in params['rerank_rt_attn_topk'].split(',')])
                else:
                    self.rerank_rt_attn_topk = None
                if 'geometric_consistency_d_thresh' in params:  # Distance threshold for building adjacency matrix for each relay token level (must be same length as RT indices)
                    self.geometric_consistency_d_thresh = tuple([float(e) for e in params['geometric_consistency_d_thresh'].split(',')])
                else:
                    self.geometric_consistency_d_thresh = (0.6,)
                if 'rerank_num_correspondences' in params:  # Total number of local correspondences for geometric consistency, per relay token/coarse feat level.
                    self.rerank_num_correspondences = tuple([int(e) for e in params['rerank_num_correspondences'].split(',')])
                else:
                    self.rerank_num_correspondences = (128,)
                if 'rerank_min_correspondences_per_window' in params:  # Minimum number of local correspondences per attention window to use for geoemetric consistency, per relay token level.
                    self.rerank_min_correspondences_per_window = tuple([int(e) for e in params['rerank_min_correspondences_per_window'].split(',')])
                else:
                    self.rerank_min_correspondences_per_window = (8,)
                self.rerank_use_attn_vals = params.getboolean('rerank_use_attn_vals', False)  # Use relay token attention values as a feature in the re-ranking classifier
                self.rerank_geotransformer_refinement = params.getboolean('rerank_geotransformer_refinement', True)  # Use geotransformer to refine local features for re-ranking
                self.rerank_num_sinkhorn_iterations = params.getint('rerank_num_sinkhorn_iterations', 100)
                self.rerank_scale_eigvec = params.getboolean('rerank_scale_eigvec', True)
                self.rerank_eigvec_layernorm = params.getboolean('rerank_eigvec_layernorm', False)
                self.rerank_output_mlp_ratio = params.getfloat('rerank_output_mlp_ratio', 1.0)
                if any(model in self.model.lower() for model in ('hotformermetricloc')):
                    #######################################################################
                    # HOTFormerMetricLoc-specific params
                    #######################################################################
                    if 'GEOTRANSFORMER' in config:
                        params = config['GEOTRANSFORMER']
                        self.geotransformer = edict()
                        self.geotransformer.disable = params.getboolean('disable', False)  # disable geotransformer for coarse feat refinement (still does LGR with raw feats)
                        self.geotransformer.input_dim = params.getint('input_dim', 2048)  # NOTE: set this to `coarse_feat_embed_dim` if used, otherwise `channels[coarse_idx]`
                        self.geotransformer.hidden_dim = params.getint('hidden_dim', 128)
                        self.geotransformer.output_dim = params.getint('output_dim', 256)
                        self.geotransformer.num_heads = params.getint('num_heads', 4)
                        if 'blocks' in params:
                            self.geotransformer.blocks = tuple([e.strip() for e in params['blocks'].split(',')])
                        else:
                            self.geotransformer.blocks = tuple(['self', 'cross', 'self', 'cross', 'self', 'cross'])
                        self.geotransformer.sigma_d = params.getfloat('sigma_d', 4.8)
                        self.geotransformer.sigma_a = params.getfloat('sigma_a', 15)
                        self.geotransformer.angle_k = params.getint('angle_k', 3)
                        self.geotransformer.activation_fn = params.get('activation_fn', 'ReLU')
                        self.geotransformer.reduction_a = params.get('reduction_a', 'max')

                    # model - Coarse Matching
                    if 'COARSE MATCHING' in config:
                        params = config['COARSE MATCHING']
                        self.coarse_matching = edict()
                        # For GT
                        self.coarse_matching.num_targets = params.getint('num_targets', 128)  # Max num coarse correspondences to consider for training
                        self.coarse_matching.overlap_threshold = params.getfloat('overlap_threshold', 0.1)  # Min overlap for selecting ground truth coarse correspondences
                        self.coarse_matching.ground_truth_matching_radius = params.getfloat('ground_truth_matching_radius', 0.6)  # Max radius to consider a GT coarse correspondence
                        # For computing coarse matching
                        self.coarse_matching.num_correspondences = params.getint('num_correspondences', 256)  # Num coarse correspondences to select
                        self.coarse_matching.dual_normalization = params.getboolean('dual_normalization', True)
                        self.coarse_matching.num_points_in_patch = params.getint('num_points_in_patch', 128)  # max num fine points to consider in patch

                    # model - Fine Matching
                    if 'FINE MATCHING' in config:
                        params = config['FINE MATCHING']
                        self.fine_matching = edict()
                        self.fine_matching.topk = params.getint('topk', 2)  # top-k potential fine correspondences to consider for each point within corresponding patches
                        self.fine_matching.acceptance_radius = params.getfloat('acceptance_radius', 0.6)  # Max radius to consider fine correspondences as inliers during LGR (maybe default to 2x voxel size?)
                        self.fine_matching.mutual = params.getboolean('mutual', True)  # Only consider mutual nearest neighbours
                        self.fine_matching.confidence_threshold = params.getfloat('confidence_threshold', 0.05)  # Min sinkhorn confidence to consider correspondence
                        self.fine_matching.use_dustbin = params.getboolean('use_dustbin', False)
                        self.fine_matching.use_global_score = params.getboolean('use_global_score', False)
                        self.fine_matching.correspondence_threshold = params.getint('correspondence_threshold', 3)  # Min num fine correspondences needed to consider a patch
                        self.fine_matching.correspondence_limit = params.getint('correspondence_limit', None)
                        self.fine_matching.num_refinement_steps = params.getint('num_refinement_steps', 5)
                        self.fine_matching.num_sinkhorn_iterations = params.getint('num_sinkhorn_iterations', 100)

            else:
                if 'ct_layers' in params:  # using carrier token attention per stage
                    self.ct_layers = tuple([e == 'True' for e in params['ct_layers'].split(',')])
                else:
                    self.ct_layers = tuple([False]*len(self.channels))

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')


class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path: str, model_params_path: str,
                 debug: bool = False, verbose: bool = False):
        """
        Configuration files

        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug
        self.verbose = verbose

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')
        # Seconday dataset for global descriptor training
        self.secondary_dataset_folder = params.get('secondary_dataset_folder', None)

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)          # Model saving frequency (in epochs)
        self.eval_freq = params.getint('eval_freq', 0)          # Model eval frequency (in epochs)
        self.embeddings_log_freq = params.getint('embeddings_log_freq', 5)  # Embeddings logging frequency (in epochs)
        self.num_embeddings_logged = params.getint('num_embeddings_logged', 20)  # Number of embeddings to log at each epoch
        self.num_workers = params.getint('num_workers', 0)
        self.wandb = params.getboolean('wandb', True)  # enable wandb logging
        self.log_grads = params.getboolean('log_grads', True)  # Log gradients (and weights) to wandb
        if 'eval_radius' in params:  # thresholds to evaluate PR at
            self.eval_radius = tuple([float(e) for e in params['eval_radius'].split(',')])
        else:
            self.eval_radius = tuple([5., 20.])
        self.finetune = params.getboolean('finetune', False)  # DEPRECATED Finetune from weights (if True, `--resume_from` resets epoch counter)
        self.finetune = False
        self.finetune_path = None  # Placeholder, value set in `trainer.py`

        # Initial batch size for global descriptors (for both main and secondary dataset)
        self.batch_size = params.getint('batch_size', 64)
        # When batch_split_size is non-zero, multistage backpropagation is enabled
        self.batch_split_size = params.getint('batch_split_size', None)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        if 'secondary_batch_size_limit' in params:
            self.secondary_batch_size_limit = params.getint('secondary_batch_size_limit')
        else:
            self.secondary_batch_size_limit = self.batch_size_limit

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        # Prioritise sampling of ground/aerial pairs, if available
        self.prioritise_cross_source = params.getboolean('prioritise_cross_source', False)
        # Only sample ground queries and aerial positives for train and val
        #   (but still considers intra-source positives/negatives within the batch)
        self.only_ground_aerial = params.getboolean('only_ground_aerial', False)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.warmup_epochs = params.getint('warmup_epochs', None)
        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                self.gamma = params.getfloat('gamma', 0.1)
                if 'scheduler_milestones' in params:
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    self.scheduler_milestones = [self.epochs+1]            
            elif self.scheduler == 'ExponentialLR':
                self.gamma = params.getfloat('gamma', 0.5)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss').lower()
        if 'contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        elif self.loss == 'truncatedsmoothap':
            # Number of best positives (closest to the query) to consider
            self.positives_per_query = params.getint("positives_per_query", 4)
            # Temperatures (annealing parameter) and numbers of nearest neighbours to consider
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)    # Margin used in loss function

        # QKV standard deviation loss (set coeffs > 0 to enable)
        self.local_qkv_std_coeff = params.getfloat('local_qkv_std_coeff', 0)
        self.rt_qkv_std_coeff = params.getfloat('rt_qkv_std_coeff', 0)
        self.qkv_target_std = params.getfloat('qkv_target_std', 1.0)  # target standard deviation for qkv projections

        # QKV weight norm loss, penalises weight collapse in QKV layers
        self.qkv_weight_norm_coeff = params.getfloat('qkv_weight_norm_coeff', 0)
        self.qkv_target_norm = params.getfloat('qkv_target_norm', 1.0)

        # Ensure only one QKV loss is chosen
        assert not ((self.local_qkv_std_coeff > 0 or self.rt_qkv_std_coeff > 0)
                    and self.qkv_weight_norm_coeff > 0), \
                        'Select either QKV std loss or QKV weight norm loss, not both'

        # Prevent QKV loss from running with multistage backprop, as it is currently not implemented correctly
        assert not ((self.local_qkv_std_coeff > 0 or self.rt_qkv_std_coeff > 0
                    or self.qkv_weight_norm_coeff > 0)
                    and self.batch_split_size not in [None, 0]), \
                        'QKV losses not compatible with multistage backprop (see L385 of trainer.py)'

        # Similarity measure: based on cosine similarity or Euclidean distance
        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        # Re-ranking
        self.rerank_loss_fn = params.get('rerank_loss_fn', None)
        if self.rerank_loss_fn is not None:
            self.rerank_loss_fn = self.rerank_loss_fn.lower()
        self.rerank_loss_coeff = params.getfloat('rerank_loss_coeff', 1.0)  # Weighting of re-ranking loss
        self.rerank_batch_size = params.getint('rerank_batch_size', None)
        if self.rerank_loss_fn is not None:
            if self.rerank_batch_size is None:
                self.rerank_batch_size = self.batch_size_limit
            assert self.rerank_batch_size <= self.batch_size_limit, 'Re-ranking batches are limited to the size of global batches'
            assert self.batch_split_size in [None, 0], 'Re-ranking loss not compatible with multistage backprop'
            # if self.batch_split_size is not None:  # REMOVED due to disabling re-ranking with multistage backprop
            #     assert self.rerank_batch_size <= self.batch_split_size, 'Can only compute re-ranking on single batch split during multi-stage backprop'
        self.rerank_enable_eval = params.getboolean('rerank_enable_eval', False)  # Optionally disable evaluating re-ranking during training (until final eval)
        self.rerank_num_neighbours = params.getint('rerank_num_neighbours', 20)  # Num neighbours to use during re-ranking (max 20 in training, unless a separate param is added to increase eval num_neighbours)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)
        self.set_aug_mode = params.getint('set_aug_mode', 1)    # Augmentation mode applied to all batch samples (1 is default)
        self.random_rot_theta = params.getfloat('random_rot_theta', 5.0)    # Random rotation (in degrees) applied during training
        self.normalize_points = params.getboolean('normalize_points', False)    # Normalize points to [-1, 1]
        self.scale_factor = params.getfloat('scale_factor', None)  # Scale factor to normalize points by a fixed scale (as done in OctFormer)
        self.unit_sphere_norm = params.getboolean('unit_sphere_norm', False)  # Use unit sphere for normalization
        self.zero_mean = params.getboolean('zero_mean', True)  # Shift point cloud to zero mean during normalization
        self.octree_depth = params.getint('octree_depth', 11)    # Set depth of octree, if octrees are used
        self.full_depth = params.getint('full_depth', 2)    # Depth of octree that is fully populated
        self.mesa = params.getfloat('mesa', 0.0)  # MESA - memory efficient sharpness optimization, enabled if > 0.0
        self.mesa_start_ratio = params.getfloat('mesa_start_ratio', 0.25)  # when to start MESA, ratio to total training time

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.validation = params.getboolean('validation', True)
        self.secondary_train_file = params.get('secondary_train_file', None)
        self.test_file = params.get('test_file', None)
        self.dataset_name = params.get('dataset_name', None)
        self.is_cross_source_dataset = 'CSWildPlaces' in self.dataset_name  # flag for dataset with ground-aerial pairs
        self.secondary_dataset_name = params.get('secondary_dataset_name', None)
        if self.secondary_dataset_name is not None:
            self.secondary_dataset_name = self.secondary_dataset_name
        self.skip_same_run = params.getboolean('skip_same_run', True)

        # If running a hyperparameter search
        self.hyperparam_search = params.getboolean('hyperparam_search', False)

        # Metric localisation and re-ranking parameters
        self.local = edict({'enable_local': False, 'max_eval_threshold': 1e10})
        if 'LOCAL' in config:
            params = config['LOCAL']
            self.local.enable_local = params.getboolean('enable_local', False)  # whether to optimise metric localisation losses
            self.local.batch_size = params.getint('local_batch_size', 8)
            self.local.aug_mode = params.getint('local_aug_mode', 1)  # Augmentation mode for local batches (1 is default)
            self.local.eval_num_workers = params.getint('eval_num_workers', 0)  # Num dataloader workers for metric loc eval dataloader (ideally higher than standard num_workers)
            self.local.max_eval_threshold = params.getfloat('max_eval_threshold', 1e10)  # max distance to NN to evaluate metric loc (prevents impossible pairs)
            self.local.icp_train = params.getboolean('icp_train', False)  # Enable icp during training (unnecessary if done during tuple creation)
            self.local.icp_eval = params.getboolean('icp_eval', False)  # Enable icp during eval
            self.local.icp_use_gicp = params.getboolean('icp_use_gicp', False)
            self.local.icp_inlier_dist_threshold = params.getfloat('icp_inlier_dist_threshold', 1.2)
            self.local.icp_max_iteration = params.getint('icp_max_iteration', 200)
            self.local.icp_voxel_size = params.getfloat('icp_voxel_size', 0.1)
            self.local.icp_two_stage = params.getboolean('icp_two_stage', False)  # Use two-stage ICP solution to fix large initial drift (e.g. in DCC)
            self.local.icp_two_stage_inlier_dist_threshold = params.getfloat('icp_two_stage_inlier_dist_threshold', 5.5)
            self.local.icp_two_stage_max_iteration = params.getint('icp_two_stage_max_iteration', 50)
            self.local.icp_two_stage_voxel_size = params.getfloat('icp_two_stage_voxel_size', 0.5)
            self.local.weight_coarse_loss = params.getfloat('weight_coarse_loss', 1.0)
            self.local.weight_fine_loss = params.getfloat('weight_coarse_loss', 1.0)
            # eval config
            self.local.acceptance_overlap = params.getfloat('acceptance_overlap', 0.0)  # min overlap to consider a ground truth coarse correspondence
            self.local.acceptance_radius = params.getfloat('acceptance_radius', 1.0)  # distance threshold for inlier fine correspondences
            self.local.inlier_ratio_threshold = params.getfloat('inlier_ratio_threshold', 0.05)  # min inlier ratio for FMR metric
            self.local.rre_threshold = params.getfloat('rre_threshold', 5.0)  # Rotation threshold to classify successful metric loc
            self.local.rte_threshold = params.getfloat('rte_threshold', 2.0)  # Translation threshold to classify successful metric loc
            # eval with ransac
            self.local.ransac_distance_threshold = params.getfloat('ransac_distance_threshold', 0.3)
            self.local.ransac_num_points = params.getint('ransac_num_points', 4)
            self.local.ransac_num_iterations = params.getint('ransac_num_iterations', 20000)

        # loss - Coarse level
        if 'COARSE LOSS' in config:
            params = config['COARSE LOSS']
            self.coarse_loss = edict()
            self.coarse_loss.positive_margin = params.getfloat('positive_margin', 0.1)
            self.coarse_loss.negative_margin = params.getfloat('negative_margin', 1.4)
            self.coarse_loss.positive_optimal = params.getfloat('positive_optimal', 0.1)
            self.coarse_loss.negative_optimal = params.getfloat('negative_optimal', 1.4)
            self.coarse_loss.log_scale = params.getfloat('log_scale', 40)
            self.coarse_loss.positive_overlap = params.getfloat('positive_overlap', 0.1)  # Min overlap to classify a positive in circle loss

        # loss - Fine level
        if 'FINE LOSS' in config:
            params = config['FINE LOSS']
            self.fine_loss = edict()
            self.fine_loss.positive_radius = params.getfloat('positive_radius', 0.6)  # Ground truth radius to consider fine correspondences

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path, local=self.local.enable_local)
        
        # Check if using octrees, load octrees instead of sparse tensor for OctFormer
        self.load_octree = any(model in self.model_params.model.lower() for model in ('octformer', 'hotformer'))

        # Ensure normalisation type is correct for octree coordinate system
        if self.load_octree and self.model_params.coordinates == 'cylindrical':
            if self.normalize_points:
                if not self.unit_sphere_norm:
                    print("[WARNING] Unit sphere normalization recommended for cylindrical octrees")
            else:
                print("[WARNING] Normalization not enabled. Ensure point clouds are already normalized within unit sphere for cylindrical octrees..")
        
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)
        if self.secondary_dataset_folder is not None:
            assert os.path.exists(self.secondary_dataset_folder), 'Cannot access secondary dataset: {}'.format(self.secondary_dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')


"""
Useful Functions
"""
def update_params_from_dict(params, param_dict: dict):    
    """
    Update training and model params from dictionary.
    """
    for key, value in param_dict.items():
        if key != 'model_params':
            setattr(params, key, value)
            continue
        for model_key, model_value in value.items():
            if model_key == 'channels_blocks_top_down_depth':
                setattr(params.model_params, 'channels', model_value[0])
                setattr(params.model_params, 'num_blocks', model_value[1])
                setattr(params.model_params, 'num_top_down', model_value[2])
                setattr(params, 'octree_depth', model_value[3])
                continue
            setattr(params.model_params, model_key, model_value)
    return params
    
def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def save_pickle(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data
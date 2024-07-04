# Warsaw University of Technology

import os
import configparser
import time
import random
import torch
import numpy as np
from ocnn.octree import Octree, Points

from dataset.quantization import PolarQuantizer, CartesianQuantizer


class ModelParams:
    def __init__(self, model_params_path):
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
        assert self.coordinates in ['polar', 'cartesian'], f'Unsupported coordinates: {self.coordinates}'

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
        self.num_top_down = params.getint('num_top_down', 1)

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
            if 'ct_layers' in params:  # using carrier token attention per stage
                self.ct_layers = tuple([e == 'True' for e in params['ct_layers'].split(',')])
            else:
                self.ct_layers = tuple([False, False, False, False])
            self.patch_size = params.getint('patch_size', 32)  # size of window attention patch
            self.dilation = params.getint('dilation', 4)  # dilation value for octree attention
            self.ct_size = params.getint('ct_size', 1)  # carrier token size, if using HAT layers
            self.ct_propagation = params.getboolean('ct_propagation', False)  # propagate ct features to local features at end of stage
            self.drop_path = params.getfloat('drop_path', 0.5)  # stochastic depth dropout
            self.input_features = params.get('input_features', 'P')  # P for global position, D for local displacement (check docs)
            self.downsample_input_embeddings = params.getboolean('downsample_input_embeddings', True)
            self.num_input_downsamples = params.getint('num_input_downsamples', 2)  # number of downsampling stages in ConvEmbed
            self.disable_RPE = params.getboolean('disable_RPE', False)
            self.conv_norm = params.get('conv_norm', 'batchnorm')  # choose normalisation layer after convolution layers
            assert self.conv_norm in ['batchnorm', 'layernorm', 'powernorm']
            self.layer_scale = params.getfloat('layer_scale', None)  # coefficient to initialise learnable channel-wise scalar multipliers for attention outputs, or None to disable this.
            self.grad_checkpoint = params.getboolean('grad_checkpoint', True)

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
    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
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

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)          # Model saving frequency (in epochs)
        self.eval_freq = params.getint('eval_freq', 0)          # Model eval frequency (in epochs)
        self.embeddings_log_freq = params.getint('embeddings_log_freq', 5)  # Embeddings logging frequency (in epochs)
        self.num_embeddings_logged = params.getint('num_embeddings_logged', 20)  # Number of embeddings to log at each epoch
        self.num_workers = params.getint('num_workers', 0)

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

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

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

        # Similarity measure: based on cosine similarity or Euclidean distance
        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)
        self.set_aug_mode = params.getint('set_aug_mode', 1)    # Augmentation mode applied to all batch samples (1 is default)
        self.normalize_points = params.getboolean('normalize_points', False)    # Normalize points to [-1, 1]
        self.octree_depth = params.getint('octree_depth', 11)    # Set depth of octree, if octrees are used
        self.full_depth = params.getint('full_depth', 2)    # Depth of octree that is fully populated
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.validation = params.getboolean('validation', True)
        self.test_file = params.get('test_file', None)
        self.dataset_name = params.get('dataset_name', None)
        self.skip_same_run = params.getboolean('skip_same_run', True)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)

        # Check if using octrees, load octrees instead of sparse tensor for OctFormer
        self.load_octree = any(model in self.model_params.model.lower() for model in ('octformer', 'hotformer'))
        
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

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
def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def set_seed(seed: int = 42):
    """
    Enable (mostly) deterministic behaviour in PyTorch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def rescale_octree_points(points: torch.Tensor, depth: int) -> torch.Tensor:
    """ 
    Rescale points stored in octree to original scale.

    Args:
        points (Tensor): Points in [0, 2^d] range, where d is octree depth.
        depth (int): Octree depth used to rescale values
    """
    # normalize points to [-1, 1] since octree points are in range [0, 2^d]
    scale = 2 ** (1 - depth)
    points_scaled = points * scale - 1.0
    return points_scaled

def octree_to_points(octree: Octree, depth: int) -> torch.Tensor:
    """
    Converts averaged points in the octree to a point cloud.

    Args:
        octree (Octree): The octree to convert to a point cloud.
        depth (int): Octree depth to query points from.
    """
    points = octree.points[depth]
    points_scaled = rescale_octree_points(points, depth)
    return points_scaled
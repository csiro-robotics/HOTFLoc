import math
import random
from typing import Optional

import numpy as np
import torch
from scipy.linalg import expm, norm
from torchvision import transforms as transforms


class TrainSetTransform:
    def __init__(self, aug_mode, random_rot_theta: float = 5.0):
        self.aug_mode = aug_mode
        self.transform = None
        if self.aug_mode == 1:
            t = [RandomRotation(max_theta=random_rot_theta, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
        elif self.aug_mode == 2:
            t = [RandomFlip([0.25, 0.25, 0.])]
        elif self.aug_mode == 0:    # No augmentations
            return None
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainTransform:
    # Augmentations for global training.
    def __init__(self, aug_mode, normalize_points=False, scale_factor=None,
                 unit_sphere_norm=False, zero_mean=True, random_rot_theta: float = 5.0):
        self.aug_mode = aug_mode
        self.normalize_points = normalize_points
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if scale_factor is not None:
            self.normalize_points = True
            self.scale_factor = scale_factor
        self.transform = None
        t = []
        # NOTE: Normalization before other augs will cause some border points to be
        #       clipped, which is fine as it is effectively additional augmentation
        if self.normalize_points:
            t.append(Normalize(scale_factor=self.scale_factor,
                               unit_sphere_norm=self.unit_sphere_norm,
                               zero_mean=self.zero_mean))
        if self.aug_mode == 1:
            # Augmentations without random rotation around z-axis (values assume [-1, 1] range)
            t.extend([JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                      RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)])
        elif self.aug_mode == 2:
            # Augmentations with random rotation around z-axis 
            t.extend([JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                      RandomRotation(max_theta=random_rot_theta, axis=np.array([0, 0, 1])),
                      RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)])
        elif self.aug_mode == 0:    # No augmentations
            pass
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        if len(t) == 0:
            return None
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class ValTransform:
    def __init__(self, normalize_points=False, scale_factor=None,
                 unit_sphere_norm=False, zero_mean=True):
        self.normalize_points = normalize_points
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if scale_factor is not None:
            self.normalize_points = True
            self.scale_factor = scale_factor
        t = None
        if self.normalize_points:
            t = Normalize(scale_factor=self.scale_factor,
                          unit_sphere_norm=self.unit_sphere_norm,
                          zero_mean=self.zero_mean)
        self.transform = t

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class Train6DOFTransform:
    # Augmentations for local training. Returns normalization
    # parameters and transformation needed to undo augmentations.
    def __init__(self, local_aug_mode, normalize_points=False, scale_factor=None,
                 unit_sphere_norm=False, zero_mean=True, random_rot_theta: float = 5.0):
        self.local_aug_mode = local_aug_mode
        self.normalize_points = normalize_points
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if scale_factor is not None:
            self.normalize_points = True
            self.scale_factor = scale_factor
        self.normalization_transform = None
        self.transform = None
        self.rotation_transform = None
        self.translation_transform = None
        t = []
        if self.normalize_points:
            self.normalization_transform = Normalize(scale_factor=self.scale_factor,
                                                     unit_sphere_norm=self.unit_sphere_norm,
                                                     zero_mean=self.zero_mean,
                                                     return_shift_and_scale=True)
        # NOTE: Not using point/block removal for local batches to simplify
        #       overlap calculation (otherwise needs to be computed on the fly)
        if self.local_aug_mode == 1:
            # Augmentations with random rotation around z-axis 
            # Note that this is in unnormalized coordinates, as opposed to the global branch transforms
            t.extend([JitterPoints(sigma=0.1)])
            self.rotation_transform = RandomRotation(max_theta=random_rot_theta,
                                                     axis=np.array([0, 0, 1]),
                                                     return_rotation=True)
            # Translation only in xy plane
            self.translation_transform = RandomTranslation(axis=np.array([1, 1, 0]),
                                                           max_delta=5,
                                                           return_translation=True)
        elif self.local_aug_mode == 0:    # No augmentations
            pass
        else:
            raise NotImplementedError('Unknown local_aug_mode: {}'.format(self.local_aug_mode))
        if len(t) == 0:
            return None
        self.transform = transforms.Compose(t)

    def __call__(self, e, ignore_rot_and_trans=False):
        shift_and_scale = None
        aug_tf = torch.eye(4)
        if self.transform is not None:
            e = self.transform(e)
        if not ignore_rot_and_trans:
            if self.rotation_transform is not None:
                e, rotation_matrix = self.rotation_transform(e)
                aug_tf[:3,:3] = torch.tensor(rotation_matrix)
            if self.translation_transform is not None:
                e, translation_vector = self.translation_transform(e)
                aug_tf[:3,-1] = torch.tensor(translation_vector)
        if self.normalization_transform is not None:
            e, shift_and_scale = self.normalization_transform(e)
        return e, shift_and_scale, aug_tf


class Val6DOFTransform:
    def __init__(self, normalize_points=False, scale_factor=None,
                 unit_sphere_norm=False, zero_mean=True):
        self.normalize_points = normalize_points
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if scale_factor is not None:
            self.normalize_points = True
            self.scale_factor = scale_factor
        self.normalization_transform = None
        if self.normalize_points:
            self.normalization_transform = Normalize(scale_factor=self.scale_factor,
                                                     unit_sphere_norm=self.unit_sphere_norm,
                                                     zero_mean=self.zero_mean,
                                                     return_shift_and_scale=True)

    def __call__(self, e, ignore_rot_and_trans=False):
        shift_and_scale = None
        aug_tf = torch.eye(4)
        if self.normalization_transform is not None:
            e, shift_and_scale = self.normalization_transform(e)
        return e, shift_and_scale, aug_tf


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=None,
                 return_rotation=False):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction
        self.return_rotation = return_rotation

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180.) * 2. * (np.random.rand(1) - 0.5))
        if self.max_theta2 is not None:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180.) * 2. * (np.random.rand(1) - 0.5))
            R = R_n @ R
        coords = coords @ R.T

        if self.return_rotation:
            return coords, R
        else:
            return coords


class RandomTranslation:
    def __init__(self, axis: Optional[np.ndarray] = None, max_delta=0.05, return_translation=False):
        self.axis = axis  # Axes to randomly translate
        if self.axis is not None:
            if self.axis.ndim == 1:
                self.axis = self.axis.reshape(1, -1)
            assert self.axis.shape == (1, 3)
        else:
            self.axis = np.ones((1, 3))
        self.max_delta = max_delta
        self.return_translation = return_translation

    def __call__(self, coords):
        trans = (self.max_delta * self.axis * np.random.randn(1, 3)).astype(np.float32)
        coords = coords + trans

        if self.return_translation:
            return coords, trans
        else:
            return coords


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords

class Normalize:
    """
    Normalize cloud within `[-norm_range, norm_range]`. Defaults to `[-1, 1]`.

    Alternatively, provide `scale_factor` to normalize the cloud with a fixed
    scaling factor. E.g. `cloud_normalized = (cloud - centroid) / scale_factor`.
    Also supports normlization within a unit sphere, and normalizing with and
    without shifting to zero mean.
    """    
    def __init__(self, norm_range: Optional[float] = None,
                 scale_factor: Optional[float] = None,
                 unit_sphere_norm: bool = False,
                 zero_mean: bool = True,
                 return_shift_and_scale: bool = False):
        assert not all([arg is not None for arg in [norm_range, scale_factor]]),\
            "Must specify one of norm_range or scale_factor, not both"
        self.norm_range = 1.0
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        self.return_shift_and_scale = return_shift_and_scale
        if norm_range is not None:
            assert norm_range > 0, "Range must be positive"
            self.norm_range = norm_range
        elif scale_factor is not None:
            assert scale_factor > 0, "Scale factor must be positive"
            self.norm_range = None
            self.scale_factor = scale_factor

    def __call__(self, coords: torch.Tensor):
        current_shift_and_scale = torch.zeros(4, dtype=coords.dtype)  # (shift_x, shift_y, shift_z, scale)
        if not self.unit_sphere_norm:                
            bbmin = coords.min(dim=0).values
            bbmax = coords.max(dim=0).values
            if self.zero_mean:
                center = (bbmin + bbmax) * 0.5
                coords = (coords - center)
                current_shift_and_scale[:3] = center
            if self.scale_factor is not None:
                coords_normalized = coords / self.scale_factor
                current_shift_and_scale[3] = self.scale_factor
            else:
                box_size = (bbmax - bbmin).max() + 1.0e-6
                current_shift_and_scale[3] = 1 / (2.0 * self.norm_range / box_size)
                coords_normalized = coords / current_shift_and_scale[3]
        else:
            # UNIT SPHERE NORMALIZATION:
            if self.zero_mean:
                centroid = torch.mean(coords, axis=0)
                coords = coords - centroid
                current_shift_and_scale[:3] = centroid
            if self.scale_factor is not None:
                max_distance = self.scale_factor
            else:
                # max_distance = torch.max(abs(coords_normalized)) / self.norm_range  ## INCORRECT, DOES NOT CONSIDER RADIAL DISTANCE
                max_distance = torch.max(torch.linalg.norm(coords, dim=1)) / self.norm_range
            coords_normalized = coords / max_distance        
            current_shift_and_scale[3] = max_distance
        
        if current_shift_and_scale[3] <= 0:
            raise ValueError("Invalid scaling factor")
        if self.return_shift_and_scale:
            return coords_normalized, current_shift_and_scale
        else:
            return coords_normalized

    @staticmethod
    def unnormalize(coords: torch.Tensor, shift_and_scale: torch.Tensor):
        """
        Undo normalization using shift and scale output from Normalize(). Note
        that this function will not verify that you have provided the correct
        shift and scale parameters for the given set of points.
        """
        assert coords.ndim == 2
        assert shift_and_scale.shape == (4,)
        if shift_and_scale[3] <= 0:
            raise ValueError("Invalid scaling factor")
        coords_unnormalized = coords * shift_and_scale[3] + shift_and_scale[:3]
        return coords_unnormalized
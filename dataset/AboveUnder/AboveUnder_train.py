# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad
# Warsaw University of Technology

import numpy as np
import torchvision.transforms as transforms

from dataset.augmentation import JitterPoints, RemoveRandomPoints, RandomTranslation, RemoveRandomBlock, RandomRotation, RandomFlip, Normalize
from dataset.base_datasets import TrainingDataset
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader


class AboveUnderTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = AboveUnderPointCloudLoader()


class TrainTransform:
    # Augmentations specific for AboveUnder dataset
    def __init__(self, aug_mode, normalize_points=False):
        self.aug_mode = aug_mode
        self.normalize_points = normalize_points
        self.transform = None
        t = []
        if self.normalize_points:
            t.append(Normalize(scale=0.95))  # [-0.95, 0.95] to prevent random translation going outside of [-1, 1]
        if self.aug_mode == 1:
            # Augmentations without random rotation around z-axis (values assume [-1, 1] range)
            t.extend([JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                      RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)])
        elif self.aug_mode == 2:
            # Augmentations with random rotation around z-axis 
            t.extend([JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                      RandomRotation(max_theta=5, axis=np.array([0, 0, 1])),
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
    # Augmentations specific for AboveUnder dataset
    def __init__(self, normalize_points=False):
        self.normalize_points = normalize_points
        t = None
        if self.normalize_points:
            t = Normalize(scale=0.95)  # [-0.95, 0.95] to prevent random translation going outside of [-1, 1]
        self.transform = t

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e
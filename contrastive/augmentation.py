import numpy as np
import torch
from torch.nn.modules.module import Module

from contrastive.augbasic import all_augmentations

class RandomAUG(Module):
    def __init__(self, configs):
        super(RandomAUG, self).__init__()

        self.p = configs.aug_p
        self.augs = all_augmentations

    def forward(self, x_torch):
        x = x_torch.clone()
        for aug in self.augs:
            if np.random.rand() < self.p:
                x = aug(x)

        return x.clone(), x_torch.clone()
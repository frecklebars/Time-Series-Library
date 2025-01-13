import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# from contrastive.augbasic import all_augmentations
from utils.augmentation import run_augmentation_single

class BasicAUG(Module):
    def __init__(self, configs):
        super(BasicAUG, self).__init__()
        self.args = configs
        self.augmentation_tags = ""

    def forward(self, x, y):
        aug_x, aug_y, self.augmentation_tags = run_augmentation_single(x, y, self.args)

        return aug_x, aug_y
    

# class RandomAUG(Module):
#     def __init__(self, configs):
#         super(RandomAUG, self).__init__()

#         self.p = configs.aug_p
#         self.augs = all_augmentations

#     def forward(self, x_torch):
#         x = x_torch.clone()
#         for aug in self.augs:
#             if np.random.rand() < self.p:
#                 x = aug(x)

#         return x.clone(), x_torch.clone()

# class AutoAUG(Module):
#     def __init__(self, configs):
#         super(AutoAUG, self).__init__()

#         self.p = configs.aug_p
#         self.augs = all_augmentations

#         self.weight = Parameter(torch.empty((2,len(self.augs))))
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)

#     def get_sampling(self, temperature=1.0, bias=0.0):
#         bias = bias + 0.0001
#         eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias) 
#         gate_inputs = torch.log(eps) - torch.log(1 - eps)
#         gate_inputs = gate_inputs.to("cuda")
#         gate_inputs = (gate_inputs + self.weight) / temperature

#         para = torch.softmax(gate_inputs, -1)
#         return para

#     def forward(self, x_torch):
#         x = x_torch.clone()
#         para = self.get_sampling() 
#         para = para.to("cuda")


#         x_aug_list = []
#         for aug in self.augs:
#             x_aug_list.append(aug(x))

#         x_aug = torch.stack(x_aug_list, 0)
#         x_aug_flat = torch.reshape(x_aug, (x_aug.shape[0], x_aug.shape[1] * x_aug.shape[2] * x_aug.shape[3]))
        
#         aug1 = torch.reshape(torch.unsqueeze(para[0], -1) * x_aug_flat, x_aug.shape)
#         aug1 = torch.sum(aug1, 0)

#         return aug1, x_torch.clone()
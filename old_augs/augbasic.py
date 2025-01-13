import torch
import torch.nn.functional as F

import numpy as np
import tsaug

# thanks https://github.com/chengw07/InfoTS

def totensor(x, device="cuda"):
    return torch.from_numpy(x).type(torch.FloatTensor).to(device)

def jitter(x, sigma=0.3):
    return x + torch.normal(mean=0., std=sigma, size=x.shape).to(x.device)

def scaling(x, sigma=0.5):
    # return x * torch.normal(mean=1., std=sigma, size=x.shape).to(x.device)
    factor = torch.normal(mean=1., std=sigma, size=(x.shape[0], x.shape[2])).to(x.device)
    res = torch.multiply(x, torch.unsqueeze(factor, 1))
    return res

def cutout(x, perc=0.1):
    seq_len = x.shape[1]
    cut_ts = x.clone()

    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len - win_len - 1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)

    cut_ts[:, start:end, :] = 0.0

    return cut_ts

def time_warp(x_torch, n_speed_change=100, max_speed_ratio=10):
    x_device = x_torch.device
    transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    x = x_torch.cpu().detach().numpy()
    x_warped = transform.augment(x)
    return totensor(x_warped.astype(np.float32), device=x_device)

def magnitude_warp(x_torch, n_speed_change=100, max_speed_ratio=10):
    x_device = x_torch.device
    transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    x = x_torch.cpu().detach().numpy()
    x_t = np.transpose(x, (0, 2, 1))
    x_warped = transform.augment(x_t).transpose((0, 2, 1))
    return totensor(x_warped.astype(np.float32), device=x_device)

def window_slice(x_torch, reduce_ratio=0.5):
    x = torch.transpose(x_torch, 2, 1)

    target_len = np.ceil(reduce_ratio * x.shape[2]).astype(int)
    start = np.random.randint(0, x.shape[2] - target_len, size=(x.shape[0])).astype(int)
    end = (start + target_len).astype(int)

    cropped_x = torch.stack([x[i, :, start[i]:end[i]] for i in range(x.shape[0])], 0)

    x_transformed = F.interpolate(cropped_x, x.shape[2], mode='linear', align_corners=False)
    x_transformed = torch.transpose(x_transformed, 2, 1)

    return x_transformed

def window_warp(x_torch, window_ratio=0.3, scales=[0.5, 2.]):
    B, T, D = x_torch.size()
    x = torch.transpose(x_torch, 2, 1)

    warp_scales = np.random.choice(scales, B)
    warp_size = np.ceil(window_ratio * T).astype(int)

    window_starts = np.random.randint(1, T - warp_size - 1, size=(B)).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    rets = []

    for i in range(x.shape[0]): #B
        window_seg = torch.unsqueeze(x[i, :, window_starts[i]:window_ends[i]], 0)
        sindow_seg_inter = F.interpolate(window_seg, int(warp_size * warp_scales[i]), mode='linear', align_corners=False)[0]

        start_seg = x[i, :, :window_starts[i]]
        end_seg = x[i, :, window_ends[i]:]

        ret_i = torch.cat([start_seg, sindow_seg_inter, end_seg], -1)
        ret_i_inter = F.interpolate(torch.unsqueeze(ret_i, 0), T, mode='linear', align_corners=False)
        rets.append(ret_i_inter)

    ret = torch.cat(rets, 0)
    ret = torch.transpose(ret, 2, 1)
    return ret

def subsequence(x):
    seq_len = x.shape[1]
    crop_len = np.random.randint(low=2, high=seq_len + 1)

    cropped_ts = x.clone()
    start = np.random.randint(seq_len - crop_len + 1)
    end = start + crop_len
    start = max(0, start)
    end = min(end, seq_len)

    cropped_ts[:, :start, :] = 0.0
    cropped_ts[:, end:, :] = 0.0

    return cropped_ts


all_augmentations = [jitter, scaling, cutout, time_warp, magnitude_warp, window_slice, window_warp, subsequence]
# all_augmentations = [jitter, scaling, cutout, time_warp, window_slice, window_warp, subsequence]
import torch
import random


def temporal_shift(x, max_frac=0.05):
    T = x.shape[1]
    k = int(max_frac * T)
    if k == 0:
        return x
    shift = random.randint(-k, k)
    return torch.roll(x, shifts=shift, dims=1)


def random_crop_start(x, min_frac=0.1, max_frac=0.3):
    T = x.shape[1]
    drop = int(random.uniform(min_frac, max_frac) * T)
    return x[:, drop:, :]


def reverse_micro_segments(x, seg_frac=0.1):
    T = x.shape[1]
    seg_len = int(seg_frac * T)
    if seg_len < 2:
        return x
    start = random.randint(0, T - seg_len)
    x = x.clone()
    x[:, start:start+seg_len, :] = torch.flip(
        x[:, start:start+seg_len, :], dims=[1]
    )
    return x


def augment(x):
    if random.random() < 0.5:
        x = temporal_shift(x)
    if random.random() < 0.5:
        x = random_crop_start(x)
    if random.random() < 0.3:
        x = reverse_micro_segments(x)
    return x


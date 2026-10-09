"""One deterministic complete pass, with explicit IDs for consumption audits."""
import math

import torch

from base_dataset import MergedCupDataset as BaseDataset


class MergedCupDataset(BaseDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        result["_coverage_index"] = int(index)
        return result


class CoverageSampler(torch.utils.data.Sampler):
    """Identical order on every rank; Accelerate then shards whole batches.

    The balancing list already contains every physical window. Only the final
    optimizer batch is padded by repeating a few shuffled entries, so there
    are no partial accumulation cycles and no dropped tail windows.
    """
    def __init__(self, size, batch, world_size, accumulation, seed=42):
        self.size, self.seed, self.epoch = size, seed, 0
        self.slots = math.ceil(size/(batch*world_size*accumulation))*(batch*world_size*accumulation)

    def order(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        return order + [order[i % self.size] for i in range(self.slots-self.size)]

    def __iter__(self):
        return iter(self.order())

    def __len__(self):
        return self.slots

    def set_epoch(self, epoch):
        self.epoch = epoch

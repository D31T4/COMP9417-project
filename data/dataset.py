import torch
from torch.utils.data import Dataset
import numpy as np
import os
import math
from collections.abc import Callable

def normalize(x: torch.Tensor, xmin: float, xmax: float):
    '''
    normalize
    '''
    return 2 * (x - xmin) / (xmax - xmin) - 1

def unnormalize(x: torch.Tensor, xmin: float, xmax: float):
    '''
    un-normalize
    '''
    return (x + 1) * (xmax - xmin) / 2. + xmin

class MotionDataset(Dataset[torch.Tensor]):
    def __init__(self, seq_len: int, fids: list[str], dir: str, memo: bool = True):
        '''
        Arguments:
        ---
        - seq_len: sequence length
        - fids: list of file ids
        - dir: directory
        - memo: load all features into memory if set to `True`
        '''
        assert fids
        assert seq_len > 0

        self.fids = fids
        self.dir = dir
        self.seq_len = seq_len

        #region read adjacency matrix
        self.adj_mat = None

        for fname in os.listdir(dir):
            if fname.endswith('.adj_mat.pt'):
                self.adj_mat: torch.Tensor = torch.load(f'{self.dir}/{fname}')
                break

        assert self.adj_mat is not None
        #endregion

        self._get_stats()

        self.memo = memo
        memo and self._mount_in_mem()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        '''
        Arguments:
        ---
        - idx: index

        Returns:
        ---
        - feat[L, V, 6]: L = `seq_len`, V = no. of joints
        '''
        i, j = self.index[idx]

        if self.memo:
            feats = self.features[i]

            # sample subsequence
            return feats[j:(j + self.seq_len)]
        else:
            coords: torch.Tensor = self._read_from_disc(i)

            # sample subsequence
            coords = coords[j:(j + 1 + self.seq_len)]

            pos = normalize(coords[1:], *self.loc_span)
            vel = normalize(coords[1:] - coords[:-1], *self.vel_span)

            return torch.cat((pos, vel), 2)
    
    def _read_from_disc(self, idx: int) -> torch.Tensor:
        '''
        read data from disc and map to features

        Arguments:
        ---
        - idx: index

        Returns:
        ---
        - feat[T, V, 3]: T = no. of frames, V = no. of joints
        '''
        return torch.load(f'{self.dir}/{self.fids[idx]}.pt')
    
    def _get_stats(self):
        '''
        get stats for normalization and build index
        '''
        loc_min = float('inf')
        loc_max = -float('inf')
        vel_min = float('inf')
        vel_max = -float('inf')

        index: list[tuple[int, int]] = []

        for fname in os.listdir(self.dir):
            if fname.endswith('.adjmat.pt'): continue

            coords: torch.Tensor = torch.load(f'{self.dir}/{fname}')
            pos = coords[1:]
            vel = coords[1:] - coords[:-1]

            loc_min = min(loc_min, pos.min().item())
            loc_max = max(loc_max, pos.max().item())
            vel_min = min(vel_min, vel.min().item())
            vel_max = max(vel_max, vel.max().item())

        for i, fid in enumerate(self.fids):
            coords: torch.Tensor = torch.load(f'{self.dir}/{fid}.pt')

            for j in range(0, coords.shape[0] - self.seq_len - 1):
                index.append((i, j))

        assert loc_min != loc_max and math.isfinite(loc_min) and math.isfinite(loc_max) and not math.isnan(loc_min) and not math.isnan(loc_max)
        assert vel_min != vel_max and math.isfinite(vel_min) and math.isfinite(vel_max) and not math.isnan(vel_min) and not math.isnan(vel_max)
        assert len(index) > 0

        self.loc_span = (loc_min, loc_max)
        self.vel_span = (vel_min, vel_max)
        self.index = index

    def _mount_in_mem(self):
        '''
        mount dataset in memory
        '''
        self.features: list[torch.Tensor] = [None] * len(self)

        for idx in range(len(self.fids)):
            coords = self._read_from_disc(idx)

            pos = normalize(coords[1:], *self.loc_span)
            vel = normalize(coords[1:] - coords[:-1], *self.vel_span)

            self.features[idx] = torch.cat((pos, vel), 2)
            
def create_split(dir: str, num_train: int, num_val: int, num_test: int, shuffle: Callable[[list[str]], None] = None):
    '''
    create a partition over motion data in directory

    Arguments:
    ---
    - dir: directory
    - num_train: size of training set
    - num_val: size of validation set
    - num_test: size of test set
    - shuffle: shuffler

    Returns:
    ---
    - train_ids: train set
    - val_ids: validation set
    - test_ids: test set
    '''
    fids = [
        fname.removesuffix('.pt') 
        for fname in os.listdir(dir) 
        if fname.endswith('.pt') and not fname.endswith('.adj_mat.pt')
    ]

    assert num_train + num_val + num_test <= len(fids)

    shuffle is not None and shuffle(fids)

    train_ids = fids[:num_train]
    val_ids = fids[num_train:(num_train + num_val)]
    test_ids = fids[(num_train + num_val):(num_train + num_val + num_test)]

    return train_ids, val_ids, test_ids

if __name__ == '__main__':
    train_ids, val_ids, test_ids = create_split('pt', 12, 4, 7, np.random.default_rng(123).shuffle)
    print(train_ids)

    dataset = MotionDataset(50, train_ids, 'pt')

    for i in range(len(dataset)):
        assert dataset[i].shape == (50, 31, 6)
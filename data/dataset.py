import torch
from torch.utils.data import Dataset
import numpy as np
import os

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
        return len(self.fids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        '''
        Arguments:
        ---
        - idx: index

        Returns:
        ---
        - feat[L, V, 6]: L = `seq_len`, V = no. of joints
        '''
        if self.memo:
            feats = self.features[idx]

            # sample subsequence
            start_idx = np.random.randint(0, feats.shape[0] - self.seq_len, dtype=int)
            return feats[start_idx:(start_idx + self.seq_len)]
        else:
            coords: torch.Tensor = self._read_from_disc(idx)

            # sample subsequence
            start_idx = np.random.randint(1, coords.shape[0] - self.seq_len, dtype=int)
            coords = coords[(start_idx - 1):(start_idx + self.seq_len)]

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
        get stats for normalization
        '''
        loc_min = float('inf')
        loc_max = -float('inf')
        vel_min = float('inf')
        vel_max = -float('inf')

        for fid in self.fids:
            coords: torch.Tensor = torch.load(f'{self.dir}/{fid}.pt')
            pos = coords[1:]
            vel = coords[1:] - coords[:-1]

            loc_min = min(loc_min, torch.min(pos))
            loc_max = max(loc_max, torch.max(pos))
            vel_min = min(vel_min, torch.min(vel))
            vel_max = max(vel_max, torch.max(vel))

        self.loc_span = (loc_min, loc_max)
        self.vel_span = (vel_min, vel_max)

    def _mount_in_mem(self):
        '''
        mount dataset in memory
        '''
        self.features: list[torch.Tensor] = [None] * len(self)

        for idx in range(len(self)):
            coords = self._read_from_disc(idx)

            pos = normalize(coords[1:], *self.loc_span)
            vel = normalize(coords[1:] - coords[:-1], *self.vel_span)

            self.features[idx] = torch.cat((pos, vel), 2)

def create_split(dir: str, num_train: int, num_val: int, num_test: int):
    '''
    create a partition over motion data in directory

    Arguments:
    ---
    - dir: directory
    - num_train: size of training set
    - num_val: size of validation set
    - num_test: size of test set

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

    np.random.shuffle(fids)

    train_ids = fids[:num_train]
    val_ids = fids[num_train:(num_train + num_val)]
    test_ids = fids[(num_train + num_val):(num_train + num_val + num_test)]

    return train_ids, val_ids, test_ids
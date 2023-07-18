import torch
from torch.utils.data import Dataset
import numpy as np
import os

def normalize(x: torch.Tensor, factor: float):
    '''
    normalize
    '''
    return x / factor

def unnormalize(x: torch.Tensor, factor: float):
    '''
    un-normalize
    '''
    return x * factor

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
            return self.features[idx]
        else:
            return self._read_from_disc(idx)
    
    def _read_from_disc(self, idx: int):
        '''
        read data from disc and map to features

        Arguments:
        ---
        - idx: index

        Returns:
        ---
        - feat[L, V, 6]: L = `seq_len`, V = no. of joints
        '''
        coords: torch.Tensor = torch.load(f'{self.dir}/{self.fids[idx]}.pt')

        # sample subsequence
        start_idx = np.random.randint(1, coords.shape[0] - self.seq_len, dtype=int)
        coords = coords[(start_idx - 1):(start_idx + self.seq_len)]

        pos = normalize(coords[1:], self.loc_span)
        vel = normalize(coords[1:] - coords[:-1], self.vel_span)

        return torch.cat((pos, vel), 2)
    
    def _get_stats(self):
        '''
        get stats for normalization
        '''
        loc_span = -float('inf')
        vel_span = -float('inf')

        for fid in self.fids:
            coords: torch.Tensor = torch.load(f'{self.dir}/{fid}.pt')
            pos = coords[1:]
            vel = coords[1:] - coords[:-1]

            loc_span = max(loc_span, torch.max(torch.abs(pos)))
            vel_span = max(vel_span, torch.max(torch.abs(vel)))

        self.loc_span = loc_span
        self.vel_span = vel_span

    def _mount_in_mem(self):
        '''
        mount dataset in memory
        '''
        self.features: list[torch.Tensor] = [None] * len(self)

        for idx in range(len(self)):
            self.features[idx] = self._read_from_disc(idx)


    

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




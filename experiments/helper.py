from data.dataset import MotionDataset, create_split
import numpy as np
import torch

DATASET_DIR = 'data/pt'

def get_dataset_path(colab: bool) -> str:
    if colab:
        return f'comp9417_proj/{DATASET_DIR}'
    else:
        return DATASET_DIR
    
def get_dataset(colab: bool):
    dir = get_dataset_path(colab)

    train_set, val_set, test_set = create_split(
        dir, 
        num_train=12, 
        num_val=4, 
        num_test=7, 
        shuffle=np.random.default_rng(123).shuffle # set seed to create same split
    )

    train_set = MotionDataset(seq_len=50, fids=train_set, dir=dir)
    val_set = MotionDataset(seq_len=50, fids=val_set, dir=dir)
    test_set = MotionDataset(seq_len=100, fids=test_set, dir=dir, memo=False)
    
    adj_mat = torch.ones((31, 31)) - torch.eye(31)
    edge_prior = torch.tensor([0.91, 0.03, 0.03, 0.03])

    return train_set, val_set, test_set, adj_mat, edge_prior
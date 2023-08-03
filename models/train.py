import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import time
import os
from collections.abc import Callable

def kl_categorial(preds: torch.Tensor, log_p: torch.Tensor, eps: float = 1e-16):
    '''
    categorical KL-divergence
    
    stolen from: https://github.com/ethanfetaya/NRI/blob/master/utils.py

    Arguments:
    ---
    - preds: predicted prob.
    - log_p: log prior
    - eps: for numerical stability

    Returns:
    ---
    - KL divergence
    '''
    ls = preds * (torch.log(preds + eps) - log_p)
    return ls.sum() / (ls.size(0) * ls.size(1))

def nll_gaussian(preds: torch.Tensor, target: torch.Tensor, variance: torch.Tensor = 5e-5):
    '''
    gaussian negative log likelihood

    stolen from: https://github.com/ethanfetaya/NRI/blob/master/utils.py

    Arguments:
    ---
    - prediction
    - target
    - variance

    Returns:
    ---
    - nll
    '''
    ls = torch.square(preds - target) / (2 * variance)
    return ls.sum() / (ls.size(0) * ls.size(2))

class CheckpointParameters:
    '''
    model checkpoint parameters
    '''
    def __init__(self, path: str, checkpt_int: int, onCheckpoint: Callable[[str], None] = None):
        '''
        Arguments:
        ---
        - path: save dir
        - checkpt_int: checkpoint interval
        - onCheckpoint: checkpoint event. called after checkpoint.
        '''
        assert os.path.isdir(path)
        self.path = path
        self.checkpt_int = checkpt_int
        self.onCheckpoint = onCheckpoint

    def checkpoint(
            self, 
            epoch: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            loss: any
        ):
        '''
        checkpoint

        Arguments:
        ---
        - epoch
        '''
        prefix = f'{self.path}/checkpt_{epoch}'

        torch.save(model.state_dict(), f'{prefix}.model.pt')
        torch.save(optimizer.state_dict(), f'{prefix}.optim.pt')
        torch.save(lr_scheduler.state_dict(), f'{prefix}.lr.pt')
        np.save(f'{prefix}.loss.npy', loss, allow_pickle=True)

        if self.onCheckpoint:
            self.onCheckpoint(prefix)
    
    def get_best_fname(self) -> str:
        '''
        get best model filename
        '''
        return f'{self.path}/best'

def train(
    model: nn.Module, 
    n_epoch: int, 
    datasets: tuple[Dataset, Dataset, Dataset], 
    edge_prior: torch.Tensor,
    train_params = None,
    checkpoint_params: CheckpointParameters = None,
    optimizer: optim.Optimizer = None,
    lr_scheduler: optim.lr_scheduler.LRScheduler = None,
    silent: bool = False,
    cuda: bool = False,
    all_train_mse: list[float] = [],
    all_train_kl: list[float] = [],
    all_train_nll: list[float] = [],
    all_valid_mse: list[float] = [],
    all_valid_kl: list[float] = [],
    all_valid_nll: list[float] = [],
    kl_coef: float = 1.
):
    '''
    train model

    Arguments:
    ---
    - model
    - n_epoch: no. of epochs
    - datasets: train set, validation set, test set
    - edge_prior: edge prior
    - train_params: training parameters
    - checkpoint_params: model checkpoint parameters
    - optimizer: optimizer
    - lr_scheduler: learning rate scheduler
    - silent: show progress bar if set to `True`
    - cuda: train on gpu
    - all_train_mse: for resume training
    - all_train_kl: for resume training
    - all_train_nll: for resume training
    - all_valid_mse: for resume training
    - all_valid_kl: for resume training
    - all_valid_nll: for resume training
    - kl_coef: set weight of kl-loss
    '''
    # current min validation loss
    current_best = float('inf')
    NLL_VAR = 5e-5

    edge_prior = torch.log(edge_prior)

    train_loader, val_loader, test_loader = [DataLoader(dataset, batch_size=8, shuffle=True) for dataset in datasets]

    if optimizer is None:
        optimizer = optim.Adam(list(model.parameters()), lr=5e-3)

    if lr_scheduler is None:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    for epoch in range(max(0, lr_scheduler.last_epoch), n_epoch):
        #region train
        train_mse: list[float] = []
        train_kl: list[float] = []
        train_nll: list[float] = []

        epoch_start = time.time()

        model.train()

        for idx, data in (pbar := tqdm(enumerate(train_loader), desc=f'[train] Epoch {epoch}', total=len(train_loader), disable=silent)):
            optimizer.zero_grad()
            
            if cuda:
                data = data.cuda()

            pred, logits = model(data, data.size(1) - 1, train_params=train_params)
            target = data[:, 1:, :, :]
            
            loss_kl = kl_categorial(F.softmax(logits, dim=-1), edge_prior)
            loss_nll = nll_gaussian(pred, target, variance=NLL_VAR)
            loss = loss_nll + kl_coef * loss_kl

            if hasattr(model.decoder, 'reg_states') and model.decoder.reg_states is not None:
                loss += model.decoder.reg_states.mean() * 1e-5

            loss.backward()
            optimizer.step()

            mse = F.mse_loss(pred, target)

            train_mse.append(mse.item())
            train_nll.append(loss_nll.item())
            train_kl.append(loss_kl.item())

            pbar.set_description(f'[train] Epoch {epoch}; NLL: {loss_nll.item():.2E}; KL: {loss_kl.item():.2E}')
            
        lr_scheduler.step()
        model.eval()

        train_kl = np.mean(train_kl)
        all_train_kl.append(train_kl)

        train_nll = np.mean(train_nll)
        all_train_nll.append(train_nll)

        train_mse = np.mean(train_mse)
        all_train_mse.append(train_mse)

        if not silent:
            print(
                '[train] Epoch: {:04d}'.format(epoch),
                '          NLL: {:.10f}'.format(train_nll),
                '           KL: {:.10f}'.format(train_kl),
                '          MSE: {:.10f}'.format(train_mse),
                '      Elapsed: {:.4f}s'.format(time.time() - epoch_start),
                sep='\n'
            )
        #endregion
        
        #region val
        epoch_start = time.time()
        
        val_mse: list[float] = []
        val_kl: list[float] = []
        val_nll: list[float] = []

        for idx, data in (pbar := tqdm(enumerate(val_loader), desc=f'[valid] Epoch {epoch}.', total=len(val_loader), disable=silent)):
            if cuda:
                data = data.cuda()

            pred, logits = model(data, data.size(1) - 1, rand=True)
            target = data[:, 1:, :, :]

            loss_nll = nll_gaussian(pred, target, variance=NLL_VAR)
            loss_kl = kl_categorial(F.softmax(logits, dim=-1), edge_prior)

            mse = F.mse_loss(pred, target)

            val_mse.append(mse.item())
            val_nll.append(loss_nll.item())
            val_kl.append(loss_kl.item())

            pbar.set_description(f'[valid] Epoch {epoch}; NLL: {loss_nll.item():.2E}; KL: {loss_kl.item():.2E}')
     
        val_kl = np.mean(val_kl)
        all_valid_kl.append(val_kl)

        val_nll = np.mean(val_nll)
        all_valid_nll.append(val_nll)

        val_mse = np.mean(val_mse)
        all_valid_mse.append(val_mse)

        if not silent:
            print(
                '[valid] Epoch: {:04d}'.format(epoch),
                '          NLL: {:.10f}'.format(val_nll),
                '           KL: {:.10f}'.format(val_kl),
                '          MSE: {:.10f}'.format(val_mse),
                '      Elapsed: {:.4f}s'.format(time.time() - epoch_start),
                sep='\n'
            )
        #endregion
        
        # checkpoint
        if checkpoint_params:
            if epoch % checkpoint_params.checkpt_int == 0:
                checkpoint_params.checkpoint(
                    epoch,
                    model,
                    optimizer,
                    lr_scheduler,
                    {
                        'train_kl': all_train_kl,
                        'train_nll': all_train_nll,
                        'train_mse': all_train_mse,
                        'valid_kl': all_valid_kl,
                        'valid_nll': all_valid_nll,
                        'valid_mse': all_valid_mse
                    }
                )

            if val_nll < current_best:
                current_best = val_nll
                torch.save(model.state_dict(), f'{checkpoint_params.get_best_fname()}.pt')

def resume(
    path: str,
    model: nn.Module, 
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    n_epoch: int, 
    datasets: tuple[Dataset, Dataset, Dataset], 
    edge_prior: torch.Tensor,
    train_params = None,
    checkpoint_params: CheckpointParameters = None,
    silent: bool = False,
    cuda: bool = False
):
    '''
    resume training.

    never used. probably not working.
    '''
    model.load_state_dict(torch.load(f'{path}.model.pt'))
    optimizer.load_state_dict(torch.load(f'{path}.optim.pt'))
    lr_scheduler.load_state_dict(torch.load(f'{path}.lr.pt'))

    all_loss = np.load(f'{path}.loss.npy', allow_pickle=True)

    all_train_mse: list[float] = all_loss['train_mse']
    all_train_kl: list[float] = all_loss['train_kl']
    all_train_nll: list[float] = all_loss['train_nll']

    all_valid_mse: list[float] = all_loss['valid_mse']
    all_valid_kl: list[float] = all_loss['valid_kl']
    all_valid_nll: list[float] = all_loss['valid_nll']

    train(
        model,
        n_epoch,
        datasets,
        edge_prior,
        train_params,
        checkpoint_params,
        optimizer,
        lr_scheduler,
        silent,
        cuda,
        all_train_mse,
        all_train_kl,
        all_train_nll,
        all_valid_mse,
        all_valid_kl,
        all_valid_nll
    )


if __name__ == '__main__':
    # run test
    import sys
    sys.path.insert(0, '..')

    from models.grand import NRI
    from data.dataset import MotionDataset, create_split

    DATASET_DIR = '../data/pt'

    train_set, val_set, test_set = create_split(
        DATASET_DIR, 
        num_train=12, 
        num_val=4, 
        num_test=7, 
        shuffle=np.random.default_rng(123).shuffle # set seed to create same split
    )

    train_set = MotionDataset(seq_len=50, fids=train_set, dir=DATASET_DIR)
    val_set = MotionDataset(seq_len=50, fids=val_set, dir=DATASET_DIR)
    test_set = MotionDataset(seq_len=100, fids=test_set, dir=DATASET_DIR, memo=False)

    adj_mat = torch.ones((31, 31)) - torch.eye(31)

    checkpt = CheckpointParameters('out/grand', 10)

    model = NRI(state_dim=6, prior_steps=50, hid_dim=64, adj_mat=adj_mat)

    edge_prior = torch.tensor([0.91, 0.03, 0.03, 0.03])

    train(model, n_epoch=2, datasets=(train_set, val_set, test_set), edge_prior=edge_prior, checkpoint_params=checkpt)
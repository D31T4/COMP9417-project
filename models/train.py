import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import time

def kl_categorial(preds: torch.Tensor, log_p: torch.Tensor, eps: float = 1e-16):
    '''
    categorical KL-divergence

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
    return ls.sum() / (preds.size(0) * preds.size(1))

def nll_gaussian(preds: torch.Tensor, target: torch.Tensor, variance: torch.Tensor = 5e-5):
    '''
    gaussian negative log likelihood

    Arguments:
    ---
    - prediction
    - target
    - variance

    Returns:
    ---
    - nll
    '''
    ls = ((preds - target) ** 2) / (2 * variance)
    return ls.sum() / (target.size(0) * target.size(1))

class CheckpointParameters:
    '''
    model checkpoint parameters
    '''
    def __init__(self, path: str, checkpt_int: int):
        '''
        Arguments:
        ---
        - path: save dir
        - checkpt_int: checkpoint interval
        '''
        self.path = path
        self.checkpt_int = checkpt_int

    def get_checkpt_fname(self, epoch: int) -> str:
        '''
        get checkpoint filename

        Arguments:
        ---
        - epoch
        '''
        return f'{self.path}/checkpt_{epoch}.pt'
    
    def get_best_fname(self) -> str:
        '''
        get best model filename
        '''
        return f'{self.path}/best.pt'

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
    debug: bool = False,
    lr: float = 5e-3
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
    '''
    # current min validation loss
    current_best = float('inf')

    edge_prior = torch.log(edge_prior)

    train_loader, val_loader, test_loader = [DataLoader(dataset, batch_size=8, shuffle=True) for dataset in datasets]

    if optimizer is None:
        optimizer = optim.Adam(list(model.parameters()), lr=lr)

    if lr_scheduler is None:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(n_epoch):
        #region train
        train_mse: list[float] = []
        train_kl: list[float] = []
        train_nll: list[float] = []

        epoch_start = time.time()

        model.train()
        optimizer.zero_grad()

        for idx, data in tqdm(enumerate(train_loader), desc=f'Epoch: {epoch}, train', total=len(train_loader), disable=silent):
            pred, logits = model(data, data.size(1) - 1, train_params=train_params)
            target = data[:, 1:, :, :]
            
            loss_kl = kl_categorial(F.softmax(logits, dim=-1), edge_prior)
            loss_nll = nll_gaussian(pred, target)
            loss = loss_nll + loss_kl

            loss.backward()
            optimizer.step()

            train_mse.append(F.mse_loss(pred, target).item())
            train_nll.append(loss_nll.item())
            train_kl.append(loss_kl.item())
            
        lr_scheduler.step()
        model.eval()
        #endregion
        
        #region val
        val_mse: list[float] = []
        val_kl: list[float] = []
        val_nll: list[float] = []

        for idx, data in tqdm(enumerate(val_loader), desc=f'Epoch: {epoch}, valid', total=len(val_loader), disable=silent):
            pred, logits = model(data, data.size(1) - 1, rand=True)
            target = data[:, 1:, :, :]

            val_mse.append(F.mse_loss(pred, target).item())
            val_nll.append(nll_gaussian(pred, target).item())
            val_kl.append(kl_categorial(F.softmax(logits, dim=-1), edge_prior).item())
        #endregion

        #region log metrics
        train_kl = np.mean(train_kl)
        train_nll = np.mean(train_nll)
        train_mse = np.mean(train_mse)
        
        val_kl = np.mean(val_kl)
        val_nll = np.mean(val_nll)
        val_mse = np.mean(val_mse)

        if not silent:
            print(
                'Epoch: {:04d}'.format(epoch),
                'nll_train: {:.10f}'.format(train_nll),
                'kl_train: {:.10f}'.format(train_kl),
                'mse_train: {:.10f}'.format(train_mse),
                'nll_val: {:.10f}'.format(val_nll),
                'kl_val: {:.10f}'.format(val_kl),
                'mse_val: {:.10f}'.format(val_mse),
                'elapsed: {:.4f}s'.format(time.time() - epoch_start),
                sep='\n'
            )
        #endregion

        # checkpoint
        if checkpoint_params:
            if epoch > 0 and not epoch % checkpoint_params.checkpt_int:
                torch.save(model.state_dict(), checkpoint_params.get_checkpt_fname(epoch))

            if val_nll < current_best:
                current_best = val_nll
                torch.save(model.state_dict(), checkpoint_params.get_best_fname())

    # TODO: output test result

if __name__ == '__main__':
    # run test
    import sys
    sys.path.insert(0, '..')

    from models.nri import NRI
    from models.grand import GraNRI
    from data.dataset import MotionDataset, create_split

    DATASET_DIR = '../data/pt'

    train_set, val_set, test_set = create_split(
        '../data/pt', 
        num_train=12, 
        num_val=4, 
        num_test=7, 
        shuffle=np.random.default_rng(123).shuffle # set seed to create same split
    )

    train_set = MotionDataset(seq_len=50, fids=train_set[0:1], dir=DATASET_DIR)
    val_set = MotionDataset(seq_len=50, fids=val_set[0:1], dir=DATASET_DIR)
    test_set = MotionDataset(seq_len=100, fids=test_set[0:1], dir=DATASET_DIR, memo=False)

    adj_mat = torch.ones((31, 31)) - torch.eye(31)

    model = GraNRI(state_dim=6, prior_steps=50, hid_dim=128, adj_mat=adj_mat)
    #edge_prior = torch.tensor([0.91, 0.03, 0.03, 0.03])
    edge_prior = torch.tensor([0.25, 0.25, 0.25, 0.25])

    train(model, n_epoch=1, datasets=(train_set, val_set, test_set), edge_prior=edge_prior, debug=True)
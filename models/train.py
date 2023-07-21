import torch
import torch.nn as nn
import torch.nn.functional as F

# use F.gumbel_softmax for reparametrization
def kl_categorial(preds: torch.Tensor, log_p: torch.Tensor, eps: float = 1e-16):
    '''
    categorical KL-divergence
    '''
    return preds * (torch.log(preds + eps) - log_p)

def loss(
    preds: torch.Tensor, 
    target: torch.Tensor,
    edge_logits: torch.Tensor,
    edge_prior: torch.Tensor,
    kl_coef: float = 1.
):
    '''
    loss function

    Arguments:
    ---
    - preds: predicted node state. tensor[B, T, V, S]
    - target: ground truth node state. tensor[B, T, V, S]
    - edge_logits: predicted edge logits.
    - edge_prior: edge prior
    - kl_coef: weight for edge classification loss

    Returns:
    ---
    - batchwise mean loss
    '''
    # regression loss
    reg = nn.BCEWithLogitsLoss(reduction='sum')(preds, target).view(preds.size(0), -1).sum(dim=1)
    
    # classification loss
    cls = kl_categorial(F.softmax(edge_logits, dim=-1), edge_prior).view(edge_logits.size(0), -1).sum(dim=1)

    ls: torch.Tensor = reg + kl_coef * cls
    return ls.mean()

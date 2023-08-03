# Models

## `nri.py`

Implements [Neural Relational Inference for Interacting Systems](https://github.com/ethanfetaya/NRI).

### Reference

- paper and code: [Neural Relational Inference for Interacting Systems](https://github.com/ethanfetaya/NRI)

### Dependencies

- torch

## `nri_PE.py`

Extends NRI with positional encoding

### Dependencies

- torch

## `grand.py`

Extends NRI with a neural PDE based decoder.

### Reference

- paper and code: [GRAND: Graph Neural Diffusion](https://arxiv.org/abs/2106.10934)

### Dependencies

- torch

- torchdiffeq (ODE solver)

## `train.py`

Code for training models.

### Dependencies

- torch
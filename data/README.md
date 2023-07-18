# Data

## `preprocess.py`

Script to pre-process raw motion data in `raw` folder into torch tensors in `pt` folder.

### Usage

1. Download `.asf` and `.amc` files from CMU Mocap database and put into `raw` folder.

2. run `python preprocess.py`.

### Dependencies

- numpy

### Reference

- data: [CMU Mocap database](http://mocap.cs.cmu.edu/)

- [documentation of .asf and .amc format](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)

- referred code: [AMCParser](https://github.com/CalciferZh/AMCParser)

## `dataset.py`

### Usage

1. Use `create_split` to split dataset into training set, validation set and test set.

2. Use `MotionDataset` to create torch dataset instance.

### Dependencies

- numpy

- torch

### Reference

- paper: [NRI](https://arxiv.org/abs/1802.04687)

- We also referred code from [dNRI](https://github.com/cgraber/cvpr_dNRI/) since NRI does not describe details on the pre-processing of motion data
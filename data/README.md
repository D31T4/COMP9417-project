# Data

## `preprocess.py`

Script to pre-process raw motion data in `raw` folder into numpy arrays in `npy` folder.

### Usage

1. Download `.asf` and `.amc` files from CMU Mocap database and put into `raw` folder.

2. run `python preprocess.py`.

### Dependencies

- numpy

### Reference

- data: [CMU Mocap database](http://mocap.cs.cmu.edu/)

- [documentation of .asf and .amc format](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)

- referred code: [AMCParser](https://github.com/CalciferZh/AMCParser)

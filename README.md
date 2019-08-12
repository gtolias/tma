# Targeted mismatch adversarial attack (TMA)

This is a Python package that uses Pytorch to implement our paper:

```
  @conference{TRC19,
   title = {Targeted Mismatch Adversarial Attack: Query with a Flower to Retrieve the Tower},
   author = {Tolias, G. and Radenovi{\'c}, F. and Chum, O.}
   booktitle = {International Conference on Computer Vision (ICCV)},
   year = {2019}
  }
  ```

It implements targeted mismatch attacks and reproduces the main experiments of the paper.

## Prerequisites

1. Python3 (tested with Python 3.5.3 on Debian 8.1)
1. PyTorch deep learning framework (tested with version 1.0.1.post2)
1. Package [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch). The code is developed with [release v1.1](https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/v1.1.tar.gz). The root folder of cnnimageretrieval-pytorch should be added to the python path 

```
export PYTHONPATH="${PYTHONPATH}:cnnimageretrieval_pytorch_1.1_rootfolder/"
```

## Usage

A simple TMA on a single image is performed by running

```
python test.py
```

All results of Table 1 in the paper are reproduced by running

```
bash  run_exp_tab1.sh
```

All results of Table 2 in the paper are reproduced by running

```
bash  run_exp_tab2.sh
```

All results of Figure 5 in the paper are reproduced by running

```
bash  run_exp_fig5.sh
```

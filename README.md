# Why Normalizing Flows Fail to Detect Out-of-Distribution Data
This repository contains experiments for the paper

_Why Normalizing Flows Fail to DetectOut-of-Distribution Data_

by  Polina Kirichenko, Pavel Izmailov and Andrew Gordon Wilson.

## Introduction

In the paper we show that the inductive biases of the flows &mdash; implicit assumptions in their architecture and training procedure &mdash; can hinder OOD detection. 
- We show that flows learn latent representations for images largely based on local pixel
  correlations, rather than semantic content, making it difficult to detect data with
  anomalous semantics.
- We identify mechanisms through which normalizing flows can simultaneously increase
  likelihood for all structured images.
- We show that by changing the architectural details of the coupling layers, we can
  encourage flows to learn transformations specific to the target data, improving OOD
  detection.
- We show that OOD detection is improved when flows are trained on high-level features
  which contain semantic information extracted from image datasets.

In this repository we provide PyTorch code for reproducing results in the paper.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/84704630-230f9d80-af28-11ea-9538-b0ea8d5d784f.png" height=200>
</p>

## Data Preparation

The following datasets need to be downloaded manually. You can then use the path to the data folder as `DATA_PATH` in the scripts below.
- CelebA: 
- NotMNIST: data available [here](https://www.kaggle.com/lubaroli/notmnist)
- ImageNet 64x64: data available [here](http://image-net.org/small/download.php); we will add ImageNet scripts soon

The other datasets can be automatically downloaded to `DATA_PATH` when you run the scripts below.

## Training RealNVP and Glow models

The scripts for training flow models are in the `experiments/train_flows/` folder.

- `train_unsup.py` &mdash; the standard script for training flows
- `train_unsup_ood.py` &mdash; same as `train_unsup.py`, but evaluates the likelihoods on OOD data during training
- `train_unsup_ood_negative.py` &mdash; same as `train_unsup_ood.py`, but minimizes likelihood on OOD data (Appendix B)
- `train_unsup_ood_uci.py` &mdash; same as `train_unsup_ood.py`, but for tabular data (Appendix K)

Comands used to train baseline models:
```bash
# RealNVP on FashionMNIST
python3 train_unsup.py --dataset=[FashionMNIST | MNIST] --data_path=DATA_PATH --save_freq=20 \
  --flow=RealNVP --logdir=LOG_DIR --ckptdir=CKPTS_DIR --num_epochs=81 --lr=5e-5 \
  --prior=Gaussian --num_blocks=6 --batch_size=32

# RealNVP on CelebA
python3 train_unsup.py --dataset=[CelebA | CIFAR10 | SVHN] --data_path=DATA_PATH --logdir=LOG_DIR \
  --ckptdir=CKPTS_DIR --num_epochs=101 --lr=1e-4 --batch_size=32 --num_blocks=8 \
  --weight_decay=5e-5 --num_scales=3

# Glow on FashionMNIST
python3 train_unsup.py --dataset=[FashionMNIST | MNIST] --data_path=DATA_PATH --flow=Glow \
  --logdir=LOG_DIR --ckptdir=CKPTS_DIR --num_epochs=151 --lr=5e-5 --batch_size=32 \
  --optim=RMSprop --num_scales=2 --num_coupling_layers_per_scale=16 \
  --st_type=highway --num_blocks=3 --num_mid_channels=200
  
# Glow on CelebA
python3 train_unsup.py --dataset=[CelebA | CIFAR10 | SVHN] --data_path=DATA_PATH --flow=Glow \
  --logdir=LOG_DIR --ckptdir=CKPTS_DIR --num_epochs=161 --lr=1e-5 --batch_size=32 \
  --optim=RMSprop --num_scales=3 --num_coupling_layers_per_scale=8 \
  --st_type=highway --num_blocks=3 --num_mid_channels=400
```


## Coupling layer and latent space visualizations

We provide example notebooks in `experiments/notebooks/`:
- `GLOW_fashion.ipynb` &mdash; Glow for FashionMNIST
- `realnvp_celeba.ipynb` &mdash; RealNVP for CelebA

Below we show latent representations learned by RealNVP trained on FashionMNIST and CelebA for in-distribution and OOD inputs.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/84704791-5c480d80-af28-11ea-822c-7d367a650c31.png" height=170>
</p>

## Negative Training

To reproduce the experiments in Appendix B, you can use the script `train_unsup_ood_negative.py`, e.g.
to maximize likelihood on CIFAR-10 and minimize likelihood on CelebA:

```bash
train_unsup_ood_negative.py --ood_dataset=CelebA --ood_data_path=OOD_DATA_PATH --dataset=CIFAR10 \
  --data_path=DATA_PATH --logdir=LOG_DIR ckptdir=CKPTS_DIR --num_epochs=101 --lr=5e-5 --batch_size=32 \
  --num_blocks=8 --num_scales=3 --negative_val=-100000 --save_freq=10 --flow=RealNVP
```

For other dataset pairs, reuse the hyper-parameters of the baseline models and set `--negative_val` equal
to `-100000` for CIFAR-10, CelebA and SVHN and to `-30000` for FashionMNIST, MNIST.


## References

The implementation of RealNVP and Glow was adapted from the [repo](https://github.com/izmailovpavel/flowgmm) for the paper _Semi-Supervised Learning with Normalizing Flows_.

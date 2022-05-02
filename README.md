# Zero-Shot Knowledge Distillation from a Decision-Based Black-Box Model

## Introduction
This is accompany code and data associated with our ICML 2021 paper.


## Requirements
PyTorch (tested on 1.9.1)

## Usage

### 1. Train a teacher model in a standard way.

python train_model_ce.py --mode teacher --dataset MNIST --architecture LeNet5

python train_model_ce.py --mode teacher --dataset FASHIONMNIST --architecture LeNet5

python train_model_ce.py --mode teacher --dataset CIFAR10 --architecture AlexNet
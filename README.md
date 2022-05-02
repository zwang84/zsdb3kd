# Zero-Shot Knowledge Distillation from a Decision-Based Black-Box Model

## Introduction
This is accompany code and data associated with our ICML 2021 paper. 

https://icml.cc/virtual/2021/poster/10257

https://arxiv.org/abs/2106.03310


## Requirements
PyTorch (tested on 1.9.1)

## Usage

### 1. Train a teacher model in a standard way.

Train a LeNet5 teacher with the MNIST dataset:
```
python train_model_ce.py --mode teacher --dataset MNIST --architecture LeNet5
```

Train a LeNet5 teacher with the FashionMNIST dataset:
```
python train_model_ce.py --mode teacher --dataset FASHIONMNIST --architecture LeNet5
```

Train a AlexNet teacher with the CIFAR10 dataset:
```
python train_model_ce.py --mode teacher --dataset CIFAR10 --architecture AlexNet
```

PS: train_model_ce.py can also be used for training/evaluating the student models (e.g., LeNet5-half, LeNet5-fifth, etc.) with the cross-entropy loss only.

### 2. Construct soft labels by calculating sample robustness with the pre-trained teacher models.

sd: sample distance; bd: boundary distance; mbd: minimal boundary distance

LeNet-5 with MNIST:
```
python get_soft_labels.py --dataset MNIST --sr_mode {sd/bd/mbd} --model_dir ./models/teacher_LeNet5_MNIST
```


LeNet-5 with FashionMNIST:
```
python get_soft_labels.py --dataset FASHIONMNIST --sr_mode {sd/bd/mbd} --model_dir ./models/teacher_LeNet5_FASHIONMNIST
```

AlexNet with CIFAR10:
```
python get_soft_labels.py --dataset CIFAR10 --sr_mode {sd/bd/mbd} --model_dir ./models/teacher_AlexNet_CIFAR10
```
## Citation
If you found this code useful, please consider citing the following work. Thank you!
```
@inproceedings{wang2021zero,
  title={Zero-shot knowledge distillation from a decision-based black-box model},
  author={Wang, Zi},
  booktitle={International Conference on Machine Learning},
  pages={10675--10685},
  year={2021},
  organization={PMLR}
}
```
This repo is under construction ...

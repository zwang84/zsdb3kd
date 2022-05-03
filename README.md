# Zero-Shot Knowledge Distillation from a Decision-Based Black-Box Model

## Introduction
This is the code and data associated with our ICML 2021 paper. 

https://icml.cc/virtual/2021/poster/10257

https://arxiv.org/abs/2106.03310


## Requirements
NumPy <br />
PyTorch (tested on 1.9.1) <br />
Torchvision <br />

## Code structure
. <br />
├── data &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # datasets downloaded or saved here <br />
├── generated_samples&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # generated pseudo samples saved here <br />
├── labels &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # generated soft labels saved here <br />
├── models &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # trained teacher models saved here <br />
├── train_model_ce.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # standard training (a teacher) with cross-entropy loss <br />
├── models.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # all model definitions and wrapper for sample robustness calculation <br />
├── get_soft_labels.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # calculate soft labels with sample robustness <br />
├── sample_robustness.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # methods for calculating sample robustness (sample distance, boundary distance, minimal boundary distance) <br />
├── train_model_kd.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # training with KD <br />
├── get_pseudo_samples.py&nbsp;&nbsp;&nbsp; # generate pseudo samples with ZSDB3KD <br />
├── untargeted_mbd.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # calculate the untargeted distances from a noise input to boundary <br />
├── README.MD &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # readme file

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

### 3. Train a student model with KD, using the generated soft labels

```
python train_model_kd.py --dataset {MNIST/FASHIONMNIST/CIFAR10} --mode {small/tiny} --logits PATH_OF_SAVED_LOGITS
```

### 4. Generate pseudo samples (ZSDB3KD)
```
python get_pseudo_samples.py --dataset MNIST --batch_size 200 --model_dir PATH_OF_SAVED_TEACHER_MODEL
```
The generated pseudo samples can be used for getting the soft labels with the 2nd and 3rd steps to test ZSDB3KD.

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

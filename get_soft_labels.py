import argparse
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader 
from torchvision.datasets import CIFAR10
from torchvision.datasets.mnist import MNIST, FashionMNIST
import torchvision.transforms as transforms

from models import ModelWrapper, LeNet5, AlexNet
from sample_robustness import SampleDist, MinimalBoundaryDist


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST", help='Dataset to be used, [MNIST, CIFAR10, Imagenet]')
parser.add_argument('--data', type=str, default='./data/', help='folder of the dataset')
parser.add_argument('--sr_mode', type=str, default='sd', help='how to calculate sample robust ness')
parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
parser.add_argument('--model_dir', type=str, default="./models/teacher_LeNet5_MNIST", help='model loading directory')
parser.add_argument('--save_folder', type=str, default="./labels/", help='soft label saved directory')
parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset == "MNIST" or args.dataset == "FASHIONMNIST":
    net = LeNet5()
    net = torch.load(args.model_dir)

    if args.dataset == "MNIST":
        data_train = MNIST(args.data,
                            download=True,
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                               ]))
        data_test = MNIST(args.data,
                          train=False,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              ]))
    else:
        data_train = FashionMNIST(args.data,
                            download=True,
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                               ]))
        data_test = FashionMNIST(args.data,
                          train=False,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              ]))

elif args.dataset == 'CIFAR10':
    net = AlexNet()
    net = torch.load(args.model_dir)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_train = CIFAR10(args.data,
                        download=True,
                        transform=transform_train)
    data_test = CIFAR10(args.data,
                        download=True,
                        train=False,
                        transform=transform_test)


def get_train_loader(batch_size, if_shuffle):
    return DataLoader(data_train, batch_size=batch_size, shuffle=if_shuffle, num_workers=8)

def get_target_train_loader(target, batch_size, if_shuffle):
    return DataLoader(sub_train_set[target], batch_size=batch_size, shuffle=if_shuffle, num_workers=8)

sub_train_loader = []
sub_train_set = []
allidx = []
for i in range(10):
    idx = np.where(np.array(data_train.targets) == i)[0].tolist()
    allidx.extend(idx)
    subset_one_class = torch.utils.data.dataset.Subset(data_train, idx)
    sub_train_set.append(subset_one_class)
    sub_train_loader.append(DataLoader(subset_one_class, batch_size=args.batch_size, shuffle=False))

net.cuda()
net.eval()

wrapped_model = ModelWrapper(net, bounds=[0,1], num_classes=10)

if args.sr_mode == 'sd':
    logits_idx = ['sample_distance']
elif args.sr_mode == 'bd':
    logits_idx = ['boundary_distance']
elif args.sr_mode == 'mbd':
    logits_idx = ['boundary_distance', 'querry1000', 'querry2000', 'querry4000', 'querry6000', 'querry8000', 
                'querry10000', 'querry12000', 'querry14000', 'querry16000', 'querry18000', 'querry20000']
    querry_idx = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
else:
    print('ERROR! invalide sr mode')
    exit()

all_soft_labels = [[] for _ in range(len(logits_idx))]
for class_idx in range(0,len(sub_train_loader)):
    print('Generating soft labels for samples from Class', class_idx)
    targets = list(range(10))
    targets.remove(class_idx)
    for i, (xi,yi) in enumerate(sub_train_loader[class_idx]):
        time_start = time.time()
        print(f"image batch: {i}")
        xi,yi = xi.cuda(), yi.cuda()

        logits = [np.zeros((xi.shape[0], 10)) for _ in range(len(logits_idx))]
        for target in targets:
            print('Calculating sample robustness against target:', target)
            if args.sr_mode == 'sd':
                sr_method = SampleDist(wrapped_model, train_dataset=get_target_train_loader(target=target, batch_size=1, if_shuffle=True))
                dist = sr_method.get_distance(xi, yi, target)
            elif args.sr_mode == 'bd':
                sr_method = MinimalBoundaryDist(wrapped_model, train_dataset=get_target_train_loader(target=target, batch_size=1, if_shuffle=True), mode = 'bd')
                dist = sr_method.get_distance(xi, yi, target)
            elif args.sr_mode == 'mbd':
                sr_method = MinimalBoundaryDist(wrapped_model, train_dataset=get_target_train_loader(target=target, batch_size=1, if_shuffle=True))
                dist = sr_method.get_distance(xi, yi, target, query_limit=20000, querry_idx=querry_idx)
            
            assert len(dist)==len(logits_idx)
            for iii in range(len(logits_idx)):
                logits[iii][:, target] = dist[iii]
        
        for iii in range(len(logits_idx)):
            logits[iii][:, class_idx] = np.sum(logits[iii], axis=1)
            all_soft_labels[iii].extend(logits[iii].tolist())
        # logit /= np.linalg.norm(logit)

        print('time used for one batch:', time.time() - time_start)
        for iii in range(len(logits_idx)):
            labels_to_save = np.array(all_soft_labels[iii])
            np.save(args.save_folder + args.dataset + '_' + args.sr_mode + '_' + logits_idx[iii] + '.npy', labels_to_save)
        print('data saved')




        





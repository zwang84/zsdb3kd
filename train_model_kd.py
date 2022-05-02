import os
from models import LeNet5Half, LeNet5Fifth, AlexNetHalf, AlexNetQuarter
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse

import scipy.special as ss
import numpy as np
import math

parser = argparse.ArgumentParser(description='train-network-knowledge-distillation')

parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','CIFAR10','CIFAR100'])
parser.add_argument('--mode', type=str, default='small', choices=['small','tiny'])
parser.add_argument('--data', type=str, default='./data/', help='folder of the dataset')
parser.add_argument('--logits', type=str, default='./labels/MNIST_sd_sample_distance.npy', help='path of the obtained pre-softmax logits file')
parser.add_argument('--temperature', type=float, default=20.0, help='temperature for kd')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print('-----------------------')
print('dataset:', args.dataset)
print('mode:', args.mode)
print('logits:', args.logits)
print('temperature:', args.temperature)
print('-----------------------')

if args.dataset == "MNIST" or args.dataset == "FASHIONMNIST":
    if args.mode == 'small':
        net = LeNet5Half().cuda()
    elif args.mode == 'tiny':
        net = LeNet5Fifth().cuda()

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

    test_loader = DataLoader(data_test, batch_size=512, num_workers=0)
    data_X = np.zeros((60000, 1, 28, 28), dtype=np.float32)
    data_y = np.zeros(60000, dtype=np.int64)
    data_softy = np.load(args.logits)
    data_softy = np.asarray(data_softy, dtype=np.float32)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
else:
    if args.mode == 'small':
        net = AlexNetHalf().cuda()
    elif args.mode == 'tiny':
        net = AlexNetQuarter().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

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
    test_loader = DataLoader(data_test, batch_size=512, num_workers=0)

    data_X = np.zeros((50000, 3, 32, 32), dtype=np.float32)
    data_y = np.zeros(50000, dtype=np.int64)
    data_softy = np.load(args.logits)
    data_softy = np.asarray(data_softy, dtype=np.float32)

count = 0
for i in range(10):
    idx = np.where(np.array(data_train.targets) == i)[0].tolist()
    subset = torch.utils.data.dataset.Subset(data_train, idx)
    subset_X = [x[0].numpy() for x in subset]
    subset_y = [x[1] for x in subset]
    subset_X = np.array(subset_X)
    subset_y = np.array(subset_y)

    data_X[count:count+len(subset_X)] = np.copy(subset_X)
    data_y[count:count+len(subset_y)] = np.copy(subset_y)
    count += len(subset_y)

# To save time, data_X and data_y could be saved as .npy files at the first time.
# Just load them afterwards
# np.save('mnist_X', data_X)
# np.save('mnist_y', data_y)
# data_X = np.load('mnist_X.npy')
# data_y = np.load('mnist_y.npy')

data_X = Variable(torch.from_numpy(data_X)).cuda()
data_y = Variable(torch.from_numpy(data_y)).cuda()
data_softy = Variable(torch.from_numpy(data_softy)).cuda()

criterion_ce = torch.nn.CrossEntropyLoss().cuda()

def shuffle_data(x, y, softy, seed=None):
    if seed is not None:
        np.random.seed(args.seed)
    idx = np.random.permutation(len(x))

    return x[idx], y[idx], softy[idx]

def train(epoch):
    global data_X, data_y, data_softy, acc_best, total_epoch
    batch_size = args.batch_size
    total_correct = 0
    net.train()
    time_start = time.time()
    for i in range(math.ceil(len(data_X)/batch_size)):
        data_X, data_y, data_softy = shuffle_data(data_X, data_y, data_softy)

        batch_X = data_X[i*batch_size:(i+1)*batch_size]
        batch_y = data_y[i*batch_size:(i+1)*batch_size]
        batch_softy = data_softy[i*batch_size:(i+1)*batch_size] / args.temperature

        optimizer.zero_grad()

        output = net(batch_X.float())
        
        pred = output.data.max(1)[1]
        total_correct += pred.eq(batch_y.data.view_as(pred)).sum()

        loss_ce = criterion_ce(output, batch_y.long())
        loss_kd = torch.nn.KLDivLoss(reduction='sum')(F.log_softmax(output / args.temperature, dim=1), F.softmax(batch_softy, dim=1))
        loss = loss_ce + loss_kd

        loss.backward()
        optimizer.step()

    train_acc = float(total_correct) / len(data_X)
    test_acc = test()
    if test_acc > acc_best:
        acc_best = test_acc
    print ("[Epoch %d/%d] [CE Loss %.4f] [KD Loss %.4f] [Total Loss %.4f] [Train acc %.4f] [Test acc %.4f] [Best acc %.4f] [Time %.1f]" 
                % (epoch, total_epoch, loss_ce.detach().cpu().numpy(), loss_kd.detach().cpu().numpy(), 
                    loss.detach().cpu().numpy(), train_acc, test_acc, acc_best, time.time()-time_start))

def test():
    global acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion_ce(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)

    return acc

if __name__ == '__main__':
    acc_best = 0
    total_epoch = 200
    for i in range(total_epoch + 1):
        train(i)

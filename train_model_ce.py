import argparse
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST, FashionMNIST
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import LeNet5, LeNet5Half, LeNet5Fifth, AlexNet, AlexNetHalf, AlexNetQuarter

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(description='train-network-cross-entropy')

parser.add_argument('--mode', type=str, default='student', choices=['teacher','student'])
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'FASHIONMNIST', 'CIFAR10','CIFAR100'])
parser.add_argument('--architecture', type=str, default='AlexNet', choices=['LeNet5','AlexNet','ResNet'])
parser.add_argument('--student_net', type=str, default='small', choices=['small','tiny'])
parser.add_argument('--data', type=str, default='./data/')
parser.add_argument('--output_dir', type=str, default='./models/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  
print('------------------------------------')
print('dataset:', args.dataset)
print('network:', args.architecture)
print('mode:', args.mode)
if args.mode == 'student':
	print('student network type:', args.student_net)
print('------------------------------------')

if args.dataset == 'MNIST' or args.dataset == 'FASHIONMNIST':
	if args.architecture != 'LeNet5':
		print('Error! MNIST or FASHIONMNIST must come with LeNet5!')
		exit()
	if args.dataset == 'MNIST':
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

	data_train_loader = DataLoader(data_train, batch_size=1024, shuffle=True, num_workers=4)
	data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)
	if args.mode == 'teacher':
		net = LeNet5().cuda()
	else:
		if args.student_net == small:
			net = LeNet5Half().cuda()
		else:
			net = LeNet5Fifth().cuda()
	
	optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
	num_samples = 60000

elif args.dataset == 'CIFAR10':
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

	data_train_loader = DataLoader(data_train, batch_size=1024, shuffle=True, num_workers=8)
	data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)

	if args.mode == 'teacher':
		net = AlexNet().cuda()
	else:
		if args.student_net == small:
			net = AlexNetHalf().cuda()
		else:
			net = AlexNetQuarter().cuda()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
	num_samples = 50000

elif args.dataset == 'FLOWERS102':
	transform_train = transforms.Compose([
		transforms.Resize(256),
		transforms.RandomResizedCrop(224),
		transforms.ToTensor(),
	])	
	transform_test = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
	])	

	data_train = datasets.ImageFolder('./data/flowers_train/', transform=transform_train)
	data_test = datasets.ImageFolder('./data/flowers_test/', transform=transform_test)

	data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=4)
	data_test_loader = DataLoader(data_test, batch_size=64, num_workers=4)

	if args.mode == 'teacher':
		net = models.resnet50(pretrained=True).cuda()
	else:
		net = models.resnet18(pretrained=True).cuda()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
	num_samples = len(data_train)

criterion = torch.nn.CrossEntropyLoss().cuda()

def adjust_learning_rate(optimizer, epoch):
	if args.architecture == 'ResNet':
		if epoch < 30:
			lr = 0.01
		elif epoch < 50:
			lr = 0.005
		else:
			lr = 0.001
	elif args.architecture == 'AlexNet':
		if epoch < 60:
			lr = 0.1
		elif epoch < 120:
			lr = 0.01
		else:
			lr = 0.001

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		
def train(epoch):
	global acc_best
	if args.dataset != 'MNIST' and args.dataset != 'FASHIONMNIST':
		adjust_learning_rate(optimizer, epoch)
	net.train()
	total_correct = 0
	total_loss = 0

	time_start = time.time()
	for i, (images, labels) in enumerate(data_train_loader):
		images, labels = Variable(images).cuda(), Variable(labels).cuda()
		optimizer.zero_grad()

		output = net(images)
		pred = output.data.max(1)[1]
		total_correct += pred.eq(labels.data.view_as(pred)).sum()

		loss = criterion(output, labels)
		total_loss += loss.data.item()

		loss.backward()
		optimizer.step()

	train_acc = float(total_correct) / num_samples
	test_acc = test()
	print ("[Epoch %d/200] [Training Loss: %.4f] [Train Acc.: %.4f] [Test Acc.: %.4f] [Time: %.2f]" 
                % (epoch, total_loss/i, train_acc, test_acc, time.time()-time_start))

	if acc_best < test_acc:
		print('found better test acc. Save!')
		if args.mode == 'teacher':
			save_name = args.output_dir + args.mode + '_' + args.architecture + '_' + args.dataset
		else:
			if args.student_net == 'small':
				save_name = args.output_dir + args.mode + '_' + args.architecture + "_small_" + args.dataset
			else:
				save_name = args.output_dir + args.mode + '_' + args.architecture + "_tiny_" + args.dataset
		torch.save(net, save_name)
		acc_best = test_acc


def test():
	net.eval()
	total_correct = 0
	with torch.no_grad():
		for i, (images, labels) in enumerate(data_test_loader):
			images, labels = Variable(images).cuda(), Variable(labels).cuda()
			output = net(images)
			pred = output.data.max(1)[1]
			total_correct += pred.eq(labels.data.view_as(pred)).sum()

	acc = float(total_correct) / len(data_test)
	return acc
 

def main():
	epoch = 200
	for e in range(1, epoch + 1):
		train(e)


if __name__ == '__main__':
	acc_best = 0
	main()

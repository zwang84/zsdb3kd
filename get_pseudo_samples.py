import argparse
import os
import time

import numpy as np
import torch

from models import ModelWrapper, LeNet5
from untargeted_mbd import UntargetedMBD


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST", help='Dataset to be used')
parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
parser.add_argument('--optimize_iter', type=int, default=40, help='iterations for optimizing the noise input')
parser.add_argument('--lbd_step', type=float, default=0.5, help='step for moving noise')
parser.add_argument('--model_dir', type=str, default="./models/teacher_LeNet5_MNIST", help='model loading directory')
parser.add_argument('--save_dir', type=str, default="./generated_samples/", help='directory to save pseudo samples')
parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset == "MNIST":
    net = LeNet5()
    net = torch.load(args.model_dir)

    n_channels = 1
    sample_size = 28
    num_classes = 10

net.cuda()
net.eval()

iter_save_idx = [5, 10, 15, 20, 25, 30, 35, 40]
pseudo_samples = {}
for i in iter_save_idx:
    pseudo_samples[i] = [[] for _ in range(num_classes)]

wrapped_model = ModelWrapper(net, bounds=[0,1], num_classes=num_classes)

for gi in range(15):
    print('Generating iteration:', gi)
    for target in range(num_classes):
        noises = []
        time_start_noise = time.time()
        print('Generating noise inputs of target class', target)
        print('This may take long if torch.rand does not set properly.')
        while True:
            scale = torch.rand(args.batch_size) 
            noise = (torch.rand((args.batch_size, n_channels, sample_size, sample_size)) + 0.2) * scale[:, np.newaxis, np.newaxis, np.newaxis]
            noise = noise.cuda()
            scale = np.random.uniform(size=args.batch_size) 
            theta = torch.randn(*noise.shape).cpu().numpy() * scale[:, np.newaxis, np.newaxis, np.newaxis]
            noise = noise + torch.tensor(theta, dtype=torch.float).cuda()

            noise = torch.clamp(noise, 0, 1).cuda()
            labels = wrapped_model.predict_label(noise, batch=True)
            idx = torch.where(labels == target)[0]

            noises.extend(noise[idx].cpu().numpy())

            if len(noises) >= args.batch_size:
                noises = torch.tensor(noises[:args.batch_size], dtype=torch.float).cuda()
                print('get all valid noise input in', time.time() - time_start_noise, 'seconds')
                break
        noises_y = wrapped_model.predict_label(noises, batch=True)
        assert torch.all(noises_y == target)

        umbd_method = UntargetedMBD(wrapped_model, train_dataset=None)
        iters = 0
        avg_dists = []
        for i in range(args.optimize_iter):
            time_start_dist = time.time()
            dists, directions = umbd_method.get_distance(noises, noises_y, query_limit=5000)
            print('time used:', time.time() - time_start_dist)
            avg_dists.append(np.mean(dists).round(3))
            print('Distances to decision boundary:', avg_dists)
            lbds = np.array([args.lbd_step for _ in range(args.batch_size)])[:, np.newaxis, np.newaxis, np.newaxis]
            new_noises = noises - torch.tensor(lbds * directions, dtype=torch.float).cuda()
            new_noises = torch.clamp(new_noises, 0, 1).cuda()

            while True:
                iter_y = wrapped_model.predict_label(new_noises, batch=True)
                idx_ok = torch.where(iter_y == target)[0]
                idx_bad = torch.where(iter_y != target)[0]

                print(np.min(lbds[:,0,0,0]))
                if len(idx_bad) > 0:
                    for i_bad in idx_bad:
                        lbds[i_bad.cpu().numpy()] = lbds[i_bad.cpu().numpy()] / 2
                        if lbds[i_bad.cpu().numpy()] < 0.1:
                            print('Warning lbd step too small:', lbds[i_bad.cpu().numpy()])
                    new_noises = noises - torch.tensor(lbds*directions, dtype=torch.float).cuda()
                    new_noises = torch.clamp(new_noises, 0, 1).cuda()
                else:
                    noises = new_noises.clone()
                    break
            if (i + 1) in iter_save_idx:
                pseudo_samples[i + 1][target].extend(noises.cpu().data.numpy())
                np.save(args.save_dir + 'class_' + str(target) + '_iter_' + str(i + 1) + '.npy', np.asarray(pseudo_samples[i + 1][target]))

'''
Codes for the calculation of boundary distance are 
extended from https://github.com/cmhcbb/attackbox.
'''

import random
import time

import numpy as np 
from numpy import linalg as LA
import torch


MAX_ITER = 1000


class UntargetedMBD(object):
    def __init__(self, model, k=200, train_dataset=None):
        self.model = model
        self.k = k
        self.train_dataset = train_dataset 
        self.log = torch.ones(MAX_ITER,2)

    def get_log(self):
        return self.log

    def get_norm(self, x):
        return LA.norm(x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]), axis=-1)

    def get_distance(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=20000,
                          distortion=None, seed=None, svm=False, momentum=0.0, stopping=0.0001):
        model = self.model
        batch_size = y0.size(0)
        query_counts = [0 for _ in range(batch_size)]
        best_thetas, g_thetas = np.zeros(x0.shape), np.array([float('inf') for _ in range(batch_size)])
        timestart = time.time()

        num_directions = 20
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        for i in range(num_directions):
            thetas = torch.zeros_like(x0).cpu().numpy()
            need_query_idx = np.arange(y0.size(0))
            while True:
                scale = np.random.uniform() * 4
                theta = torch.randn(x0[0].shape) * scale       #theta shape:(1,1,28,28) or (1,3,32,32)

                new_targets = model.predict_label(x0[need_query_idx]+theta.cuda(), batch=True)
                idx_good = torch.where(new_targets != y0[0].item())[0]
                idx_bad = torch.where(new_targets == y0[0].item())[0]
                if len(idx_good) > 0:
                    updated_idx = need_query_idx[idx_good.cpu().numpy()]
                    thetas[updated_idx] = theta.cpu().numpy()
                    need_query_idx = np.delete(need_query_idx, idx_good.cpu().numpy())
                if len(need_query_idx) == 0:
                    break
                
            initial_lbds = self.get_norm(thetas)

            thetas = thetas / initial_lbds[:, np.newaxis, np.newaxis, np.newaxis]
            lbds, counts = self.fine_grained_binary_search(model, x0, y0, thetas, initial_lbds, g_thetas)

            update_idx = np.where(lbds < g_thetas)[0]
            g_thetas[update_idx] = lbds[update_idx]
            best_thetas[update_idx,:,:,:] = thetas[update_idx,:,:,:]

        timeend = time.time()

        print("==========> Found best average distance %.4f in %.4f seconds using %.2f mean queries" %
              (np.mean(g_thetas), timeend-timestart, np.mean(query_counts)))
        
        xg, gg = best_thetas, g_thetas
        for i in range(iterations):
            sign_gradients, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

            # Line search
            ls_counts = np.array([0 for _ in range(y0.size(0))])
            min_thetas = xg
            min_g2s = gg

            for _ in range(15):
                new_thetas = xg - alpha * sign_gradients

                new_thetas_mag = LA.norm(new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]), axis=-1)
                new_thetas = new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]) / new_thetas_mag[:,np.newaxis]
                new_thetas = new_thetas.reshape(xg.shape)

                new_g2s, counts = self.fine_grained_binary_search_local(model, x0, y0, new_thetas, initial_lbd = min_g2s, tol=beta/500)
                ls_counts += counts
                alpha = alpha * 2

                new_g2s_small_idx = np.where(new_g2s < min_g2s)[0]

                # may need more optimization on new_g2 < min_g2 here
                if len(new_g2s_small_idx) > 0:
                    min_thetas[new_g2s_small_idx] = new_thetas[new_g2s_small_idx]
                    min_g2s[new_g2s_small_idx] = new_g2s[new_g2s_small_idx]
                else:
                    break
            
            min_g2s_bigger_idx = np.where(min_g2s >= gg)[0]

            if len(min_g2s_bigger_idx) != 0 and len(min_g2s_bigger_idx) != len(min_g2s):
                print('Error on if min_g2 >= gg')
                exit()

            if len(min_g2s_bigger_idx) == len(min_g2s):
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_thetas = xg - alpha * sign_gradients
                    new_thetas_mag = LA.norm(new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]), axis=-1)
                    new_thetas = new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]) / new_thetas_mag[:,np.newaxis]
                    new_thetas = new_thetas.reshape(xg.shape)

                    new_g2s, counts = self.fine_grained_binary_search_local(model, x0, y0, new_thetas, initial_lbd = min_g2s, tol=beta/500)
                    ls_counts += counts

                    new_g2s_small_idx = np.where(new_g2s < gg)[0]

                    if len(new_g2s_small_idx) > 0:
                        min_thetas[new_g2s_small_idx] = new_thetas[new_g2s_small_idx]
                        min_g2s[new_g2s_small_idx] = new_g2s[new_g2s_small_idx]
                        break

            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break

            xg, gg = min_thetas, min_g2s
            query_counts += (grad_queries + ls_counts)

            if np.mean(query_counts) > query_limit:
                print('reach query limit break')
                break

            if (i+1)%5==0:
                print("Iteration %3d distance %.4f num_queries %d" % (i+1, np.mean(gg), np.mean(query_counts)))
        return gg, xg


    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquerys = np.array([0 for _ in range(y0.size(0))])
        lbds = initial_lbd
        lbd_los = np.zeros_like(initial_lbd)
        lbd_his = np.copy(initial_lbd)

        adds = theta * lbds[:, np.newaxis, np.newaxis, np.newaxis]
        results = model.predict_label(x0 + torch.tensor(adds, dtype=torch.float).cuda(), batch=True)
        nquerys = [x+1 for x in nquerys]

        not_target_idx = np.where(results.cpu() == y0[0].item())[0]
        target_idx = np.where(results.cpu() != y0[0].item())[0]

        if len(not_target_idx) > 0:
            lbd_his[not_target_idx] = lbds[not_target_idx] * 1.01
            current_not_target_idx = np.copy(not_target_idx)

            while len(current_not_target_idx) > 0:
                add_his = theta * lbd_his[:, np.newaxis, np.newaxis, np.newaxis]
                results = model.predict_label(x0[current_not_target_idx] + torch.tensor(add_his[current_not_target_idx], dtype=torch.float).cuda(), batch=True)
                for i in current_not_target_idx:
                    nquerys[i] += 1
                current_not_target_idx = current_not_target_idx[np.where(results.cpu() == y0[0].item())[0]]
                if len(current_not_target_idx) > 0:
                    lbd_his[current_not_target_idx] = lbd_his[current_not_target_idx] * 1.01
                    g100_idx = current_not_target_idx[np.where(lbd_his[current_not_target_idx] > 20)[0]]
                    if len(g100_idx) > 0:
                        current_not_target_idx = np.setdiff1d(current_not_target_idx,g100_idx)

        if len(target_idx) > 0:
            lbd_los[target_idx] = lbds[target_idx] * 0.99
            current_target_idx = np.copy(target_idx)
            while True:
                add_los = theta * lbd_los[:, np.newaxis, np.newaxis, np.newaxis]
                results = model.predict_label(x0[current_target_idx] + torch.tensor(add_los[current_target_idx], dtype=torch.float).cuda(), batch=True)
                for i in current_target_idx:
                    nquerys[i] += 1
                current_target_idx = current_target_idx[np.where(results.cpu() != y0[0].item())[0]]
                if len(current_target_idx) > 0:
                    lbd_los[current_target_idx] = lbd_los[current_target_idx] * 0.99
                else:
                    break
        
        lbd_mids = np.zeros_like(lbd_his)
        need_binary_idx = np.where((lbd_his - lbd_los) > tol)[0]

        while len(need_binary_idx) > 0:
            lbd_mids[need_binary_idx] = (lbd_los[need_binary_idx] + lbd_his[need_binary_idx]) / 2.0
            
            for i in need_binary_idx:
                nquerys[i] += 1
            add_mids = theta * lbd_mids[:,np.newaxis, np.newaxis, np.newaxis]
            results = model.predict_label(x0[need_binary_idx] + torch.tensor(add_mids[need_binary_idx], dtype=torch.float).cuda(), batch=True)
            
            not_target_idx = need_binary_idx[np.where(results.cpu() == y0[0].item())[0]]
            target_idx = need_binary_idx[np.where(results.cpu() != y0[0].item())[0]]

            if len(target_idx) > 0:
                lbd_his[target_idx] = lbd_mids[target_idx]
            if len(not_target_idx) > 0:
                lbd_los[not_target_idx] = lbd_mids[not_target_idx]

            need_binary_idx = np.where((lbd_his - lbd_los) > tol)[0]

        return lbd_his, nquerys

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)   #[batch_size,1,28,28]
        queries = [0 for _ in range(y0.size(0))]

        for iii in range(K):
            t00=time.time()

            u = torch.randn(*theta.shape).cpu().numpy()
            u_mag = LA.norm(u.reshape(u.shape[0],u.shape[1]*u.shape[2]*u.shape[3]), axis=-1)
            u = u.reshape(u.shape[0],u.shape[1]*u.shape[2]*u.shape[3]) / u_mag[:,np.newaxis] * h
            u = u.reshape(theta.shape)

            signs = np.array([1 for _ in range(y0.size(0))])

            new_theta = theta + u
            new_theta_mag = LA.norm(new_theta.reshape(new_theta.shape[0],new_theta.shape[1]*new_theta.shape[2]*new_theta.shape[3]), axis=-1)
            new_theta = new_theta.reshape(new_theta.shape[0],new_theta.shape[1]*new_theta.shape[2]*new_theta.shape[3]) / new_theta_mag[:,np.newaxis]
            new_theta = new_theta.reshape(theta.shape)
            
            new_adds = new_theta.reshape(new_theta.shape[0],new_theta.shape[1]*new_theta.shape[2]*new_theta.shape[3]) * np.array(initial_lbd)[:,np.newaxis]
            new_adds = new_adds.reshape(theta.shape)

            results = self.model.predict_label(x0+torch.tensor(new_adds, dtype=torch.float).cuda(),batch=True)
            

            reverse_sign_idx = np.where(results.cpu() != y0[0].item())[0]
            signs[reverse_sign_idx] = -1
            
            queries = [x+1 for x in queries]

            u = u.reshape(u.shape[0],u.shape[1]*u.shape[2]*u.shape[3]) * signs[:,np.newaxis]
            u = u.reshape(theta.shape)

            sign_grad += u
        sign_grad /= K

        return sign_grad, queries

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquerys = [0 for _ in range(y0.size(0))]

        need_query_idx = np.where(initial_lbd > current_best)[0]
        no_need_query = []

        if need_query_idx.size != 0:
            need_query_adds = theta.reshape(theta.shape[0],theta.shape[1]*theta.shape[2]*theta.shape[3]) * np.array(current_best)[:,np.newaxis]
            need_query_adds = need_query_adds.reshape(theta.shape)[need_query_idx]
            results = model.predict_label(x0[need_query_idx] + torch.tensor(need_query_adds, dtype=torch.float).cuda(),batch=True)
            no_need_query = need_query_idx[np.where(results.cpu() == y0[0].item())[0]]
            for i in need_query_idx:
                nquerys[i] += 1

        lbd = [current_best[i] if i in no_need_query else initial_lbd[i] for i in range(y0.size(0))]

        lbd_hi = np.array(lbd)
        lbd_lo = np.array([0.0 for _ in range(y0.size(0))])

        while len(no_need_query) < y0.size(0):
            need_query = np.delete(list(range(y0.size(0))), np.array(no_need_query).astype(int))
            lbd_mid = np.array([(lbd_lo[i] + lbd_hi[i])/2.0 for i in range(y0.size(0))])

            for i in need_query:
                nquerys[i] += 1
            need_query_adds = theta.reshape(theta.shape[0],theta.shape[1]*theta.shape[2]*theta.shape[3])[need_query] * np.array(lbd_mid[need_query])[:,np.newaxis]
            need_query_adds =  need_query_adds.reshape(len(need_query), theta.shape[1], theta.shape[2], theta.shape[3])
            results = model.predict_label(x0[need_query] + torch.tensor(need_query_adds, dtype=torch.float).cuda(),batch=True)

            change_lo_idx = need_query[np.where(results.cpu() == y0[0].item())[0]]
            change_hi_idx = need_query[np.where(results.cpu() != y0[0].item())[0]]

            if len(change_lo_idx) > 0:
                lbd_lo[change_lo_idx] = lbd_mid[change_lo_idx]
            if len(change_hi_idx) > 0:
                lbd_hi[change_hi_idx] = lbd_mid[change_hi_idx]
            
            need_delete = need_query[np.where(np.array([(lbd_hi[i] - lbd_lo[i]) < 1e-5 for i in need_query]) == True)[0]]
            no_need_query = np.concatenate((no_need_query,need_delete))

        return lbd_hi, nquerys   













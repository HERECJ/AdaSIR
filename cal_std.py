import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



import scipy as sp
import scipy.sparse as ss
import scipy.io as sio

# import logging
# import torch
# import math


class Eval:
    @staticmethod
    def evaluate_item(train:ss.csr_matrix, test:ss.csr_matrix, user:np.ndarray, item:np.ndarray, topk:int=50, cutoff:int=50, mode:int=0):
        pop_count = train.sum(axis=0).A
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75

        
        user_count = train.sum(axis=1).A

        train = train.tocsr()
        test = test.tocsr()
        idx = np.squeeze((test.sum(axis=1) > 0).A)
        train = train[idx, :]
        test = test[idx, :]
        user = user[idx, :]
        N = train.shape[1]
        cand_count = N - train.sum(axis=1)
        # std_res = Eval.compute_std(train, test, user, item)
        return Eval.compute_std(train, test, user, item, pop_count)
    
    @staticmethod
    def compute_std(train_mat:ss.csr_matrix, test_mat:ss.csr_matrix, user:np.ndarray, item:np.ndarray, pop_count:np.ndarray):
        M, N = train_mat.shape
        res_arr = np.zeros((M, 4))
        # mean for fn, std for fn, mean for tn, std for tn
        # fn:false negative, tn:true negative
        pop_items = (-pop_count).argsort()

        for i in range(M):
            ratings = np.matmul(user[i], item.T)
            idices_arr = np.zeros(N)
            idices_arr[pop_items[:100]] = np.inf
            fn_items = test_mat.indices[test_mat.indptr[i]:test_mat.indptr[i + 1]]
            train_items = train_mat.indices[train_mat.indptr[i]:train_mat.indptr[i + 1]]
            idices_arr[fn_items] = -np.inf
            idices_arr[train_items] = -np.inf
            fn_score = ratings[fn_items]
            tn_score = ratings[idices_arr > 0]
            res_arr[i,0] = fn_score.mean()
            res_arr[i,1] = fn_score.std()
            res_arr[i,2] = tn_score.mean()
            res_arr[i,3] = tn_score.std()
        
        return res_arr


class Eval_Std:
    @staticmethod
    def evaluate_item(train, test, user_emb, item_emb, item_emb_std, sample_cnt=1000, uid=0):
        user_pop_count = train.sum(axis=1).A

        train = train.tocsr()
        test = test.tocsr()
        idx = np.squeeze((test.sum(axis=1) > 0).A)
        train = train[idx, :]
        test = test[idx, :]
        user_emb = user_emb[idx, :]
        
        return Eval_Std.compute_std(train, test, user_emb, item_emb, item_emb_std, user_pop_count, sample_cnt=sample_cnt, uid=uid)


    
    def compute_std(train_mat, test_mat, user_emb, item_emb, item_emb_std, user_pop_count, sample_cnt=1000, uid=0):
        M, N = train_mat.shape
        indices_arr = np.ones(N)
        pop_users = (-user_pop_count).argsort()
        

        u = pop_users[uid][0]
        user_emb_u = user_emb[u]

        fn_items = test_mat.indices[test_mat.indptr[u]: test_mat.indptr[u + 1]]
        train_items = train_mat.indices[train_mat.indptr[u] : train_mat.indptr[u + 1]]

        indices_arr[fn_items] = -np.inf
        indices_arr[train_items] = -np.inf

        fn_num = len(fn_items)
        tn_num = len(indices_arr[indices_arr>0])
        # 采样很多次计算方差
        rating_mtx = np.zeros((sample_cnt, N))
        rat_to_frame_fn = np.zeros((sample_cnt * fn_num, 3))
        rat_to_frame_tn = np.zeros((sample_cnt * tn_num, 3))
        for i in range(sample_cnt):
            sampled_item_emb = Eval_Std.reparameterize(item_emb, item_emb_std)
            ratings = np.matmul(user_emb_u, sampled_item_emb.T)
            rating_mtx[i] = ratings

            # rat_to_frame_fn[i * fn_num : (i + 1)*fn_num, 0] = np.ones(fn_num) * i
            # rat_to_frame_fn[i * fn_num : (i + 1)*fn_num, 1] = np.arange(fn_num)
            # rat_to_frame_fn[i * fn_num : (i + 1)*fn_num, 2] = ratings[fn_items]

            # rat_to_frame_tn[i * tn_num : (i + 1)*tn_num, 0] = np.ones(tn_num) * i
            # rat_to_frame_tn[i * tn_num : (i + 1)*tn_num, 1] = np.arange(tn_num)
            # rat_to_frame_tn[i * tn_num : (i + 1)*tn_num, 2] = ratings[indices_arr > 0]
            
            
        rating_std = rating_mtx.std(axis=0)
        rating_mean = rating_mtx.mean(axis=0)
        return rating_std[fn_items], rating_std[indices_arr > 0], rating_mean[fn_items], rating_mean[indices_arr > 0], rat_to_frame_fn, rat_to_frame_tn


    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

if __name__ == "__main__":
    model = torch.load('model_ts_bpr.pkl')
    user_emb = model._User_Embedding.weight.cpu().data
    item_emb = model._Item_Embedding.weight.cpu().data
    item_emb_std = model._Item_Embedding_std.weight.cpu().data

    sample_num = 20000
    uid = 0

    from dataloader import RecData, UserItemData
    data = RecData('datasets','ml100k')
    train_mat, test_mat = data.get_data(0.8)
    evals = Eval_Std()
    fn, tn, fn_mean, tn_mean, rating_mtx_fn, rating_mtx_tn = evals.evaluate_item(train_mat, test_mat, user_emb, item_emb, item_emb_std, sample_num, uid)
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.set()
    sns.kdeplot(fn, shade=True, label='False Negative')
    sns.kdeplot(tn, shade=True, label='True Negative')

    plt.xlabel('Std w.r.t. Predited Rating')
    plt.legend()

    plt.subplot(1,2,2)
    sns.set()
    sns.kdeplot(fn_mean, shade=True, label='False Negative')
    sns.kdeplot(tn_mean, shade=True, label='True Negative')
    plt.xlabel('Mean w.r.t. Predited Rating')
    plt.legend()
    
    file_name = 'fig/std_mean_user%d_cnt%d.png'%(uid, sample_num)
    plt.savefig(file_name)



    # plt.figure(figsize=(6.5,6))
    # sns.set()
    # import pandas as pd
    # data = pd.DataFrame(rating_mtx_fn, columns=['num','item','rate'])
    # data2  = pd.DataFrame(rating_mtx_tn, columns=['num','item','rate'])
    

    # sns.relplot(x='item',y='rate', data=data, kind="line", label='False Negative')
    # # sns.relplot(x='item',y='rate', data=data2, kind="line", label='True Negative')

    # file_name = 'fig/test.png'
    # plt.savefig(file_name)
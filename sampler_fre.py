import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class base_sampler(nn.Module):
    """
    Uniform sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(base_sampler, self).__init__()
        self.num_items = num_items
        self.num_neg = num_neg
        self.device = device
    
    def update_pool(self, user_embs, item_embs, **kwargs):
        pass
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.num_neg), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.num_neg, device=self.device))

class two_pass(nn.Module):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.sample_size = sample_size # importance sampling
        self.pool_size = pool_size # resample
        self.num_neg = num_neg # negative samples
        self.device = device
        
        self.pool = torch.zeros(num_users, pool_size, device=device, dtype=torch.long)

        # self.weigts_sample = torch.ones(self.pool_size, device=device)
    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))

    
    def re_sample(self, user_batch, user_embs_batch, item_embs, neg_items, log_neg_q):
        pred = (user_embs_batch.unsqueeze(1) * item_embs[neg_items]).sum(-1) - log_neg_q
        sample_weight = F.softmax(pred, dim=-1)
        idices = torch.multinomial(sample_weight, self.pool_size, replacement=True)
        return torch.gather(neg_items, 1, idices), torch.gather(pred, 1, idices)
    
    # @profile
    def __update_pool__(self, user_batch, tmp_pool, tmp_score):
        idx = self.pool[user_batch].sum(-1) < 1
        
        user_init = user_batch[idx]
        self.pool[user_init] = tmp_pool[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        # weights = self.weigts_sample.repeat(num_user_update, 2)
        # idx_k = torch.multinomial(weights, self.pool_size, replacement=True)
        idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
        candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
        self.pool[user_update] = torch.gather(candidate, 1, idx_k)
    
    def update_pool(self, user_embs, item_embs, batch_size=2048, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
            user_embs_batch = user_embs[user_batch]
    
            neg_items, neg_q = self.sample_Q(user_batch)
            tmp_pool, tmp_score = self.re_sample(user_batch, user_embs_batch, item_embs, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score)
    
    # @profile
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        
        # idx_no_unitial = candidates.sum(-1) < 1
        # if user_id[idx_no_unitial].shape[0] > 0 :
        #     print(user_id[idx_no_unitial],candidates)
        #     import pdb; pdb.set_trace()
        #     raise ValueError('No candidate items!!!\n Please initialize the sample pool')
        
        # idx_k = torch.multinomial(self.weigts_sample.repeat(batch_size, 1), self.num_neg, replacement=True)
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)

        return torch.gather(candidates, 1, idx_k), -torch.log(self.pool_size * torch.ones(batch_size, self.num_neg, device=self.device))

class two_pass_weight(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass_weight, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score):
        idx = self.pool[user_batch].sum(-1) < 1
        
        user_init = user_batch[idx]
        if len(user_init) > 0:
            self.pool[user_init] = tmp_pool[idx]
            self.pool_weight[user_init] = tmp_score[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        # weights = self.weigts_sample.repeat(num_user_update, 2)
        # idx_k = torch.multinomial(weights, self.pool_size, replacement=True)
        if num_user_update > 0:
            idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
            candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
            candidate_weight = torch.cat([self.pool_weight[user_update], tmp_score[~idx]], dim=1)
            self.pool[user_update] = torch.gather(candidate, 1, idx_k)
            self.pool_weight[user_update] = torch.gather(candidate_weight, 1, idx_k).detach()
    
    def forward(self, user_id, random_flag=True, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        candidates_weight = self.pool_weight[user_id]

        # random
        if random_flag is True:
            idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        else:
        # ranking
            idx_k = torch.multinomial(candidates_weight, self.num_neg, replacement=True)
        return torch.gather(candidates, 1, idx_k), -torch.log(torch.gather(candidates_weight, 1, idx_k))


class two_pass_weight_pop(two_pass_weight):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, mat, mode=0, **kwargs):
        super(two_pass_weight_pop, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
        pop_count = torch.squeeze(torch.from_numpy((mat.sum(axis=0).A).astype(np.float32)).to(device))
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / torch.sum(pop_count)
    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        items = torch.multinomial(self.pop_prob.repeat(batch_size, 1), self.sample_size, replacement=True)
        # return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))
        # import pdb; pdb.set_trace()
        return items, torch.log(self.pop_prob[items])
    
    



class two_pass_is(nn.Module):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass_is, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.sample_size = sample_size
        self.pool_size = pool_size
        self.num_neg = num_neg
        self.device = device

        self.is_pool = torch.zeros(self.num_users, self.sample_size, device=device, dtype=torch.long)
        self.is_pool_weight = torch.zeros(self.num_users, self.sample_size, device=device, dtype=torch.float)

        self.user_pool = torch.zeros(self.num_users, self.pool_size, device=device, dtype=torch.long)
        self.weigts_sample = torch.ones(self.pool_size, device=device)
        self.weigts_sample_is = torch.ones(self.sample_size, device=device)
    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))

    def re_sample(self, user_batch, user_embs_batch, item_embs, neg_items, log_neg_q):
        pred = (user_embs_batch.unsqueeze(1) * item_embs[neg_items]).sum(-1) - log_neg_q
        # tmp_weight = pred - log_neg_q
        idx = self.is_pool[user_batch].sum(-1) == 0
        user_init = user_batch[idx]
        if len(user_init) > 0:
            self.is_pool[user_init] = neg_items[idx]
            self.is_pool_weight[user_init] = pred[idx]
            # sample_weight = torch.softmax(self.is_pool_weight, dim=-1)
            k = torch.multinomial(F.softmax(pred[idx], dim=-1), self.pool_size, replacement=True)
            
            self.user_pool[user_init] = torch.gather(neg_items[idx], 1, k)
        
        user_update = user_batch[~idx]
        if len(user_update) > 0:
            # www = torch.cat([pred[~idx], self.is_pool_weight[user_update]], dim=1)
            www = torch.cat([self.is_pool_weight[user_update], pred[~idx]], dim=1)
            k = torch.multinomial(F.softmax(www, dim=-1), self.pool_size, replacement=True)
            # candidates = torch.cat([neg_items[~idx], self.is_pool[user_update]], dim=1)
            candidates = torch.cat([self.is_pool[user_update], neg_items[~idx]], dim=1)
            self.user_pool[user_update] = torch.gather(candidates, 1, k)


            weight = self.weigts_sample_is.repeat(user_update.shape[0], 2)
            idx_k = torch.multinomial(weight, self.sample_size, replacement=True)
            # import pdb; pdb.set_trace()
            self.is_pool[user_update] = torch.gather(candidates, 1, idx_k)
            # self.ttmp = torch.gather(www, 1, idx_k)
            # self.is_pool_weight[user_update] = self.ttmp
            self.is_pool_weight[user_update] = torch.gather(www, 1, idx_k).detach()
            
    
    def update_pool(self, user_embs, item_embs, batch_size=2048, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
            user_embs_batch = user_embs[user_batch]

            neg_items, neg_q = self.sample_Q(user_batch)
            self.re_sample(user_batch, user_embs_batch, item_embs, neg_items, neg_q)

        
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.user_pool[user_id]

        # idx_no_unitial = candidates.sum(-1) < 1
        # if user_id[idx_no_unitial].shape[0] > 0 :
        #     print(user_id[idx_no_unitial],candidates)
        #     raise ValueError('No candidate items!!!\n Please initialize the sample pool')
    
        idx = torch.multinomial(self.weigts_sample.repeat(batch_size, 1), self.num_neg, replacement=True)
        return torch.gather(candidates, 1, idx), -torch.log(self.pool_size * torch.ones(batch_size, self.num_neg, device=self.device))



def plot_probs(orig_dis, count_freq):
    import matplotlib.pyplot as plt
    orig_cum = np.cumsum(orig_dis)
    count_cum = np.cumsum(count_freq)
    plt.plot(orig_cum, label='orig')
    plt.plot(count_cum, label='sample')
    plt.legend()
    plt.savefig('fig/1.jpg')

if __name__ == '__main__':
    import numpy as np 
    device = torch.device('cuda')
    user_num = 63690
    # user_num = 917
    item_num = 8939
    sample_size = 200
    pool_size = 50
    sample_num = 5

    dim = 32

    epoch = 200

    user_emb = torch.randn(user_num, dim, device=device)
    item_emb = torch.randn(item_num, dim, device=device)

    user_id = 0
    sampler = two_pass(user_num, item_num, sample_size, pool_size, sample_num, device)
    # sampler = two_pass_is(user_num, item_num, sample_size, pool_size, sample_num, device)
    # sampler = base_sampler(user_num, item_num, sample_size, pool_size, sample_num, device)

    pred = (user_emb[user_id].repeat(item_num,1) * item_emb).sum(-1)
    orig_dis = torch.softmax(pred, 0).cpu().numpy()
    # orig_dis = np.cumsum(prob)


    count_arr = np.zeros(item_num)
    import time
    t0 = time.time()
    for ii in range(epoch):
        # print(ii)
        sampler.update_pool(user_emb, item_emb)
        user_id = torch.LongTensor([user_id])
        
        neg_id, _ = sampler(user_id)
        uniq_idx, uniq_count = np.unique(neg_id.cpu().numpy(), return_counts=True)
        # import pdb; pdb.set_trace()
        # print(uniq_idx, uniq_count)
        for i, item in enumerate(uniq_idx):
            count_arr[uniq_idx[i]] += uniq_count[i]
    
    # import pdb; pdb.set_trace()
    count_freq = count_arr / count_arr.sum()
    t1 = time.time()
    print("running time", t1 - t0)

    # plot_probs(orig_dis, count_freq)
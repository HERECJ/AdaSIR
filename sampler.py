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

class base_sampler_pop(base_sampler):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, mat, mode=0, **kwargs):
        super(base_sampler_pop, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)

        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
        pop_count = torch.squeeze(torch.from_numpy((mat.sum(axis=0).A).astype(np.float32)).to(device))
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / torch.sum(pop_count)
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        min_batch_size = min(4096, batch_size)
        cnt = batch_size // min_batch_size
        if (batch_size - cnt * min_batch_size) > 0:
            cnt += 1
        items = torch.zeros(batch_size, self.num_neg, dtype=torch.long, device=self.device)
        items_prob = torch.zeros(batch_size, self.num_neg, device=self.device)
        for c in range(cnt):
            end_index = min((c+1)*min_batch_size, batch_size)
            mmmm = end_index - c * min_batch_size
            items_min_batch = torch.multinomial(self.pop_prob.repeat(mmmm,1), self.num_neg)
            items_prob_min_batch = torch.log(self.pop_prob[items_min_batch])
            
            items[c * min_batch_size : end_index] = items_min_batch
            items_prob[c * min_batch_size : end_index] = items_prob_min_batch

        return items, items_prob



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

    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))

    
    def re_sample(self, user_batch, user_embs_batch, item_embs, neg_items, log_neg_q):
        pred = (user_embs_batch.unsqueeze(1) * item_embs[neg_items]).sum(-1) - log_neg_q
        sample_weight = F.softmax(pred, dim=-1)
        idices = torch.multinomial(sample_weight, self.pool_size, replacement=True)
        return torch.gather(neg_items, 1, idices), torch.gather(sample_weight, 1, idices)
    
    # @profile
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag=False):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            return

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
    
    def update_pool(self, user_embs, item_embs, batch_size=2048, cover_flag=False, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
            user_embs_batch = user_embs[user_batch]
    
            neg_items, neg_q = self.sample_Q(user_batch)
            tmp_pool, tmp_score = self.re_sample(user_batch, user_embs_batch, item_embs, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)
    
    # @profile
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), -torch.log(self.pool_size * torch.ones(batch_size, self.num_neg, device=self.device))

class two_pass_pop(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, mat, mode=0, **kwargs):
        super(two_pass_pop, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)

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
        items = torch.multinomial(self.pop_prob.repeat(batch_size,1), self.sample_size)
        return items, torch.log(self.pop_prob[items])

class two_pass_rank(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.candidate_items = torch.zeros(num_users, sample_size, device=self.device, dtype=torch.long)
    
    def update_pool(self, user_embs, item_embs, batch_size=2048, cover_flag=False, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
            user_embs_batch = user_embs[user_batch]
    
            neg_items, neg_q = self.sample_Q(user_batch)
            self.candidate_items[user_batch] = neg_items
            tmp_pool, tmp_score = self.re_sample(user_batch, user_embs_batch, item_embs, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)
    
class two_pass_discount(two_pass):
    """
        Update the pool with time discount
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.pool_cnt = torch.zeros(num_users, pool_size, device=self.device)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag, tao=0.5):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            self.pool_cnt[user_batch] = self.pool_cnt[user_batch].detach() + 1
            return
        
        idx = self.pool[user_batch].sum(-1) < 1
        user_init = user_batch[idx]
        self.pool[user_init] = tmp_pool[idx]
        self.pool_cnt[user_init] = self.pool_cnt[user_init].detach() + 1

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]

        candidates = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
        new_pool_cnt = torch.ones(num_user_update, self.pool_size, device=self.device)
        candidate_pool_cnt = torch.cat([self.pool_cnt[user_update] + 1, new_pool_cnt], dim=1)

        sample_prob = torch.exp(torch.negative(tao * candidate_pool_cnt))  #可以修改这边采样的分布
        idx_k = torch.multinomial(sample_prob, self.pool_size, replacement=True)
        # idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
        self.pool[user_update] = torch.gather(candidates, 1, idx_k)
        self.pool_cnt[user_update] = torch.gather(candidate_pool_cnt, 1, idx_k).detach()
        return 


class two_pass_weight(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass_weight, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag=False):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            self.pool_weight[user_batch] = tmp_score.detach()
            return

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
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        candidates_weight = self.pool_weight[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
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
        return items, torch.log(self.pop_prob[items])

class two_pass_weight_rank(two_pass_weight):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.candidate_items = torch.zeros(num_users, sample_size, device=self.device, dtype=torch.long)
    
    def update_pool(self, user_embs, item_embs, batch_size=2048, cover_flag=False, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
            user_embs_batch = user_embs[user_batch]
    
            neg_items, neg_q = self.sample_Q(user_batch)
            self.candidate_items[user_batch] = neg_items
            tmp_pool, tmp_score = self.re_sample(user_batch, user_embs_batch, item_embs, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)


class tapast(base_sampler):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.pool_size = pool_size
        self.num_users = num_users

    def update_pool(self, user_embs, item_embs, cover_flag=False, **kwargs):
        # self.pool = torch.randint(0, self.num_items, size=(self.num_users, self.pool_size), device=self.device)
        pass

    def forward(self, user_id, model=None, **kwargs):
        batch_size = user_id.shape[0]
        pool = torch.randint(0, self.num_items, size=(batch_size, self.num_neg, self.pool_size), device=self.device)
        rats = model.inference(user_id.repeat(self.num_neg, 1).T, pool)
        r_v, r_idx = rats.max(dim=-1)
        return r_idx, torch.exp(r_v)



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

    epoch = 10000

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

    plot_probs(orig_dis, count_freq)
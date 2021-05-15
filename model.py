import torch
import torch.nn as nn
import torch.nn.functional as F


# class MF_Rec(nn.Module):
#     def __init__(self, num_item, num_user, dims, *args, **kwargs):
#         super(MF_Rec, self).__init__()
#         self.num_item = num_item
#         self.num_user = num_user
#         self.dims = dims
        
#         self._User_Emb = nn.Embedding(self.num_user, self.dims)
#         self._Item_Emb = nn.Embedding(self.num_item, self.dims)
        
#     def forward(self, user_id, pos_id, neg_id):
#         user_emb = self._User_Emb(user_id)
#         pos_item_emb = self._Item_Emb(pos_id)
#         neg_item_emb = self._Item_Emb(neg_id)
#         return (user_emb * pos_item_emb).sum(-1), (user_emb * neg_item_emb).sum(-1)
    
#     def loss_function(self, pos_rat, pos_prob, neg_rat, neg_prob, reduction=False, mode=0):
#         '''
#             mode : {0:pair-wise(BPR), 1:list-wise(Softmax)}
#         '''
#         if mode == 0:


class BaseModel(nn.Module):
    def __init__(self, num_user, num_item, dims, loss_mode=0, **kwargs):
        """
            loss_mode : 0 for pair-wise loss, 1 for softmax loss
        """
        super(BaseModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dims = dims
        self.loss_mode = loss_mode
        # self.mm = nn.LogSigmoid()

    def forward(self, user_id, pos_id, neg_id):
        pass
    

    def kl_loss(self, mu, log_var, anneal=1.0, reduction=False):
        pass

    def loss_function(self, neg_rat, neg_prob, pos_rat, pos_prob=None, reduction=False, weighted=False, **kwargs):
        """
            reduction : False the sum over the entries. True the average value
        """
        if self.loss_mode == 0:
            return self.pair_wise_loss(pos_rat, neg_rat, neg_prob, reduction=reduction, weighted=weighted)
        elif self.loss_mode == 1:
            return self.softmax_loss(pos_rat, pos_prob, neg_rat, neg_prob, reduction=reduction, weighted=weighted)
        else:
            raise ValueError('%d, Not supported loss mode'%self.loss_mode)
    
    def pair_wise_loss(self, pos_rat, neg_rat, neg_prob, reduction=False, weighted=False):
        # shape of pos_rat (B,), neg_rat (B,M)
        pred = torch.subtract(pos_rat.unsqueeze(1), neg_rat)
        
        if weighted:
            # 这边计算有问题，和原来的不一样，这里面用的是基于重要性采样的，但是实际上应该用采样器返回的概率
            importance = F.softmax(torch.negative(pred) - neg_prob, dim=1)
        else:
            importance = F.softmax(torch.ones_like(pred), dim=1)
        weight_loss = torch.multiply(importance.detach(), torch.negative(F.logsigmoid(pred)))
        if reduction:
            return torch.sum(weight_loss, dim=-1).mean(-1)
        else:
            return torch.sum(weight_loss, dim=-1).sum(-1)
        
    
    def softmax_loss(self, pos_rat, pos_prob, neg_rat, neg_prob, reduction=False):
        idx_mtx = (pos_rat != 0).double()
        new_pos = pos_rat - torch.log(pos_prob.detach())
        new_neg = neg_rat - torch.log(neg_prob.detach())
        parts_log_sum_exp = torch.logsumexp(new_neg, dim=-1).unsqueeze(-1)
        final = torch.log( torch.exp(new_pos) + torch.exp(parts_log_sum_exp))
        if reduction is True:
            return torch.sum((- new_pos + final) * idx_mtx, dim=-1 ).mean()
        else:
            return torch.sum((- new_pos + final) * idx_mtx, dim=-1 ).sum()


class BaseMF(BaseModel):
    def __init__(self, num_user, num_item, dims, loss_mode=0, **kwargs):
        # assert loss_mode==0, 'Only supported pair-wise loss for MF models'
        super(BaseMF, self).__init__(num_user, num_item, dims, loss_mode)
        self._User_Embedding = nn.Embedding(self.num_user, self.dims)
        # nn.init.normal_(self._User_Embedding.weight, mean=0, std=0.1)
        self._Item_Embedding = nn.Embedding(self.num_item, self.dims)
        # nn.init.normal_(self._Item_Embedding.weight, mean=0, std=0.1)

    def forward(self, user_id, pos_id, neg_id):
        '''
        user_id: (B,)
        pos_id:  (B,)
        neg_id:  (B,M) M is the number of negative samples
        '''
        user_emb = self._User_Embedding(user_id)
        pos_emb = self._Item_Embedding(pos_id)
        neg_emb = self._Item_Embedding(neg_id)
        return (user_emb * pos_emb).sum(-1), (user_emb.unsqueeze(1) * neg_emb).sum(-1)
    
    def get_user_embs(self, eval_flag=True):
        return self._User_Embedding.weight
    
    def get_item_embs(self,  eval_flag=True):
        return self._Item_Embedding.weight
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        return (user_emb.unsqueeze(-2) * item_emb).sum(-1)


class BaseMF_TS(BaseMF):
    def __init__(self, num_user, num_item, dims, loss_mode=2, **kwargs):
        assert loss_mode==2
        super(BaseMF_TS, self).__init__(num_user, num_item, dims, loss_mode)
        self._User_Embedding = nn.Embedding(self.num_user, self.dims)
        # self._User_Embedding_std = nn.Embedding(self.num_user, self.dims)
        
        self._Item_Embedding = nn.Embedding(self.num_item, self.dims)
        self._Item_Embedding_std = nn.Embedding(self.num_item, self.dims)
    
    def forward(self, user_id, pos_id, neg_id):
        # user_emb = self.reparameterize(self._User_Embedding(user_id), self._User_Embedding_std(user_id))
        user_emb = self._User_Embedding(user_id)

        pos_emb = self.reparameterize(self._Item_Embedding(pos_id), self._Item_Embedding_std(pos_id))
        neg_emb = self.reparameterize(self._Item_Embedding(neg_id), self._Item_Embedding_std(neg_id))
        # pos_emb = self._Item_Embedding(pos_id)
        # neg_emb = self._Item_Embedding(neg_id)

        return (user_emb * pos_emb).sum(-1), (user_emb.unsqueeze(1) * neg_emb).sum(-1)

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def kl_loss(self, mu, log_var, anneal=1.0, std_p=1.0, reduction=False):
        if reduction is True:
            return -anneal * 0.5 * torch.mean(torch.sum(1 + log_var - 2 * torch.log(std_p) - mu.pow(2) - log_var.exp()/std_p.pow(2), dim = 1), dim = 0)
        else:
            return -anneal * 0.5 * torch.sum(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
    

    def loss_function(self, neg_rat, neg_prob, pos_rat, pos_prob=None, reduction=False, weighted=False, anneal=0.01, std_p=0.001, **kwargs):
        """
            reduction : False the sum over the entries. True the average value
        """
        batch_size = pos_rat.shape[0]
        loss0 = self.pair_wise_loss(pos_rat, neg_rat, neg_prob, reduction=reduction, weighted=weighted)
        # klloss = self.kl_loss(self._User_Embedding.weight, self._User_Embedding_std.weight, reduction=reduction) + self.kl_loss(self._Item_Embedding.weight, self._Item_Embedding_std.weight, reduction=reduction)
        klloss = self.kl_loss(self._Item_Embedding.weight, self._Item_Embedding_std.weight, anneal=anneal, std_p=std_p, reduction=False)
        return loss0 , klloss

    def get_user_embs(self, eval_flag=True):
        # if eval_flag is True:
            # return self._User_Embedding.weight
        # else:
            # return self.reparameterize(self._User_Embedding.weight, self._User_Embedding_std.weight)
        return self._User_Embedding.weight

    def get_item_embs(self, eval_flag=True):
        if eval_flag is True:
            return self._Item_Embedding.weight
        else:
            return self.reparameterize(self._Item_Embedding.weight, self._Item_Embedding_std.weight)
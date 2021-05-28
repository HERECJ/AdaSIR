from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_user, num_item, dims, **kwargs):
        """
            loss_mode : 0 for pair-wise loss, 1 for softmax loss
        """
        super(BaseModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dims = dims

    def forward(self, user_id, pos_id, neg_id):
        pass
    

    def loss_function(self, neg_rat, neg_prob, pos_rat, reduction=False, weighted=False, pos_rank=None, **kwargs):
        """
        Bpr loss
        """
        pred = torch.subtract(pos_rat.unsqueeze(1), neg_rat)
        
        if weighted:
            importance = F.softmax(torch.negative(pred) - neg_prob, dim=1)
        else:
            importance = F.softmax(torch.ones_like(pred), dim=1)

        if pos_rank is not None:
            importance = importance * pos_rank

        weight_loss = torch.multiply(importance.detach(), torch.negative(F.logsigmoid(pred)))
        if reduction:
            return torch.sum(weight_loss, dim=-1).mean(-1)
        else:
            return torch.sum(weight_loss, dim=-1).sum(-1)

    def inference(self, user_id, item_id):
        pass

    def est_rank(self, user_id, pos_rat, candidate_items, sample_size):
        # (rank - 1)/N = (est_rank - 1)/sample_size
        candidate_rat = self.inference(user_id.repeat(candidate_items.shape[1],1).T, candidate_items)
        sorted_seq, _ = torch.sort(candidate_rat)
        quick_r = torch.searchsorted(sorted_seq, pos_rat.unsqueeze(-1))
        r = ((quick_r) * (self.num_item - 1 ) / sample_size ).floor().long()
        return self._rank_weight_pre[r]
    
    def cal_n(self,n):
        vec = 1 / torch.arange(0,n)
        return vec[1:].sum()
    


class BaseMF(BaseModel):
    def __init__(self, num_user, num_item, dims, pos_weight=False,**kwargs):
        super(BaseMF, self).__init__(num_user, num_item, dims)
        
        self._User_Embedding = nn.Embedding(self.num_user, self.dims) 
        self._Item_Embedding = nn.Embedding(self.num_item, self.dims)
        if pos_weight is True:
            self._rank_weight_pre = torch.tensor([self.cal_n(x) if x > 1 else 1 for x in range(self.num_item + 1)])
        self._init_emb()
    
    def _init_emb(self):
        nn.init.normal_(self._User_Embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self._Item_Embedding.weight, mean=0, std=0.1)
        
    
    def get_user_embs(self, eval_flag=True):
        return self._User_Embedding.weight
    
    def get_item_embs(self,  eval_flag=True):
        return self._Item_Embedding.weight
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        return (user_emb * item_emb).sum(-1)
    
    # def forward(self, user_id, pos_id, neg_id):
    #     pos_rat = self.inference(user_id, pos_id)
    #     neg_rat = self.inference(user_id.unsqueeze(1), neg_id)
    #     import pdb; pdb.set_trace()
    #     return pos_rat, neg_rat
    def forward(self, user_id, pos_id, neg_id):
        N = neg_id.shape[1]
        pos_rat = self.inference(user_id, pos_id)
        neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
        return pos_rat, neg_rat
    
    # def cal_rank(self, user_id, pos_rat):
    #     item_emb = self._Item_Embedding.weight
    #     user_emb = self._User_Embedding(user_id)
    #     all_rat = torch.matmul(user_emb, item_emb.T)
    #     sorted_seq, _ = torch.sort(all_rat)
    #     cal_r = torch.searchsorted(sorted_seq, pos_rat.unsqueeze(-1))
    #     return cal_r

class NCF(BaseMF):
    def __init__(self, num_user, num_item, dims, pos_weight=False, **kwargs):
        super().__init__(num_user, num_item, dims, pos_weight=pos_weight, **kwargs)
        self._FC = nn.Linear(dims, 1, bias=False)
        self._W = nn.Linear(2 * dims, dims)
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        gmf_out = user_emb * item_emb
        mlp_out = self._W(torch.cat([user_emb, item_emb], dim=-1))
        inferences = self._FC(torch.tanh(gmf_out + mlp_out))
        return inferences.squeeze(-1)
    
    # def forward(self, user_id, pos_id, neg_id):
    #     N = neg_id.shape[1]
    #     pos_rat = self.inference(user_id, pos_id)
    #     neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
    #     return pos_rat, neg_rat

class MLP(BaseMF):
    def __init__(self, num_user, num_item, dims, pos_weight=False, **kwargs):
        super().__init__(num_user, num_item, dims, pos_weight=pos_weight, **kwargs)
        self._FC = nn.Linear(dims, 1, bias=False)
        self._W = nn.Linear(2 * dims, dims)
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        mlp_out = self._W(torch.cat([user_emb, item_emb], dim=-1))
        inferences = self._FC(torch.tanh(mlp_out))
        return inferences.squeeze(-1)
    
    # def forward(self, user_id, pos_id, neg_id):
    #     N = neg_id.shape[1]
    #     pos_rat = self.inference(user_id, pos_id)
    #     neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
    #     return pos_rat, neg_rat

class GMF(BaseMF):
    def __init__(self, num_user, num_item, dims, pos_weight=False, **kwargs):
        super().__init__(num_user, num_item, dims, pos_weight=pos_weight, **kwargs)
        self._FC = nn.Linear(dims, 1, bias=False)
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        gmf_out = user_emb * item_emb
        inferences = self._FC(gmf_out)
        return inferences.squeeze(-1)
    
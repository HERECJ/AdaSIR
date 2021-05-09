import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from two_pass_sampler import two_phrases_sampler
from sampler import base_sampler, two_pass, two_pass_weight, two_pass_weight_pop
from model import BaseMF, BaseModel, BaseMF_TS
from dataloader import RecData, UserItemData
import argparse
import numpy as np
from utils import Eval
from cal_std import Eval as Eval2
import utils
import logging
import scipy as sp
import scipy.io
import datetime
import time
import math
import os


def evaluate(model, train_mat, test_mat, config, logger, device):
    logger.info("Start evaluation")
    model.eval()
    with torch.no_grad():
        user_num, item_num = train_mat.shape
        
        user_emb = model.get_user_embs().cpu().data
        item_emb = model.get_item_embs().cpu().data
        
        users = np.random.choice(user_num, min(user_num, 5000), False)
        evals = Eval()
        m = evals.evaluate_item(train_mat[users, :], test_mat[users, :], user_emb[users, :], item_emb, topk=50)
        
        eeee = Eval2()
        nn = eeee.evaluate_item(train_mat[users, :], test_mat[users, :], user_emb[users, :], item_emb)
    return m, nn



# @profile
def train_model(model, sampler, train_mat, test_mat, config, logger):
    optimizer = utils_optim(config, model)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)
    device = torch.device(config.device)
    # sampler = two_phrases_sampler(user_emb,item_emb,config.sample_size,config.pool_size,config.num_neg)

    for epoch in range(config.epoch):
        sampler.zero_grad()
        if epoch % config.update_epoch < 1:
            user_emb = model.get_user_embs(eval_flag=False)
            item_emb = model.get_item_embs(eval_flag=False)
            sampler.update_pool(user_emb, item_emb)
        # print(sampler.is_pool, sampler.is_pool_weight)
        # print(sampler.is_pool.shape, sampler.is_pool_weight.shape)

        loss_ = 0.0
        kl_loss_ = 0.0
        logger.info("Epoch %d"%epoch)
    
        train_data = UserItemData(train_mat)
        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True)

        for batch_idx, data in enumerate(train_dataloader):
            model.train()
            sampler.train()
            user_id, item_id = data
            user_id, item_id = user_id.to(device), item_id.to(device)
            optimizer.zero_grad()
            
            neg_id, prob_neg = sampler(user_id, config.random_flag)
            pos_rat, neg_rat = model(user_id, item_id, neg_id) 

            loss, kl = model.loss_function(neg_rat, prob_neg, pos_rat, reduction=config.reduction, weighted=config.weighted, anneal=config.anneal)
            
            # if (batch_idx % 20) == 0:
                # logger.info("--Batch %d, loss : %.4f"%(batch_idx, loss.data))
            loss_ += loss
            kl_loss_ += kl
            (loss + kl/math.exp(2 * epoch + 1)).backward()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        
        logger.info('--loss : %.2f, --kl loss: %.2f '% (loss_, kl_loss_))
            
        scheduler.step()

        if (epoch % 10) == 0:
            result, res_std = evaluate(model, train_mat, test_mat, config, logger, device)
            logger.info('***************Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(result['item_ndcg'][4], result['item_ndcg'][9], result['item_ndcg'][49]))
            logger.info('***************Eval_Res : RECALL@5,10,50 %.6f, %.6f, %.6f'%(result['item_recall'][4], result['item_recall'][9], result['item_recall'][49]))
            # print(res_std.mean(0))
            logger.info(res_std.mean(0))


def utils_optim(config, model):
    if config.optim=='adam':
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optim=='sgd':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError('Unkown optimizer!')
        

def main(config, logger=None):
    device = torch.device(config.device)
    data = RecData(config.data_dir, config.data)
    train_mat, test_mat = data.get_data(config.ratio)
    user_num, item_num = train_mat.shape
    logging.info('The shape of datasets: %d, %d'%(user_num, item_num))
    
    # model = BaseMF(user_num, item_num, config.dims)
    model = BaseMF_TS(user_num, item_num, config.dims)
    sampler_list = [base_sampler, two_pass, two_pass_weight, two_pass_weight_pop]
    assert config.sampler < 4, ValueError("Not supported sampler")
    if config.sampler < 3:
        sampler = sampler_list[config.sampler](user_num, item_num, config.sample_size, config.pool_size, config.sample_num, device)
    else:
        sampler = sampler_list[config.sampler](user_num, item_num, config.sample_size, config.pool_size, config.sample_num, device, train_mat)

    model = model.to(device)
    sampler = sampler.to(device)
    train_model(model, sampler, train_mat, test_mat, config, logger)
    torch.save(model, 'model_ts_bpr.pkl')

    return evaluate(model, train_mat, test_mat, config, logger, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('--data', default='ml100k', type=str, help='path of datafile')
    parser.add_argument('-d', '--dims', default=32, type=int, help='the dimenson of the latent vector for student model')
    parser.add_argument('-s','--sample_num', default=5, type=int, help='the number of sampled items')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=100, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='the learning rate for training')
    parser.add_argument('--seed', default=10, type=int, help='random seed values')
    parser.add_argument('--ratio', default=0.8, type=float, help='the spilit ratio of dataset for train and test')
    parser.add_argument('--log_path', default='logs_test_test', type=str, help='the path for log files')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for dataloader')
    parser.add_argument('--data_dir', default='datasets', type=str, help='the dir of datafiles')
    parser.add_argument('--device', default='cuda', type=str, help='device for training, cuda or gpu')
    parser.add_argument('--sampler', default=0, type=int, help='the sampler, 0 : uniform, 1 : two pass, 2 : two pass with weight')
    parser.add_argument('--fix_seed', action='store_true', help='whether to fix the seed values')
    parser.add_argument('--step_size', default=50, type=int, help='step size for learning rate discount') 
    parser.add_argument('--gamma', default=0.95, type=float, help='discout for lr')
    parser.add_argument('--reduction', default=False, type=bool, help='loss if reduction')
    parser.add_argument('--sample_size', default=200, type=int, help='the number of samples for importance sampling')
    parser.add_argument('--pool_size', default=50, type=int)
    parser.add_argument('--update_epoch', default=1, type=int, help='the intervals to update the sample pool')
    parser.add_argument('--weighted', action='store_true', help='whether weighted for the loss function')
    parser.add_argument('--random_flag', action='store_true', help='uniform sample from the pool or multinomial')
    parser.add_argument('--anneal', default=0.001, type=float, help='the coefficient for the KL loss')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay for the optimizer')



    config = parser.parse_args()

    import os
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((config.data, str(config.sampler), str(config.sample_size), str(config.pool_size), str(config.sample_num),timestamp))
    log_file_name = os.path.join(config.log_path, loglogs)
    logger = utils.get_logger(log_file_name)
    
    logger.info(config)
    if config.fix_seed:
        utils.setup_seed(config.seed)
    m, _ = main(config, logger)
    # print('ndcg@5,10,50, ', m['item_ndcg'][[4,9,49]])

    logger.info('Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(m['item_ndcg'][4], m['item_ndcg'][9], m['item_ndcg'][49]))
    logger.info('Eval_Res : RECALL@5,10,50 %.6f, %.6f, %.6f'%(m['item_recall'][4], m['item_recall'][9], m['item_recall'][49]))

    logger.info("Finish")
    svmat_name = log_file_name + '.mat'
    scipy.io.savemat(svmat_name, m)

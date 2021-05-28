#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"

dataset='gowalla'
logs='log_gowalla'
# dataset='yelp'
n=5
epoch=200
batch=4096


for lr in 0.01
do
    for w in 0.01 0.001 0.0001
    do
        for m in 0 1
        do
            python main_more.py -m ${m} --data ${dataset} --sampler 0 -s $n -e ${epoch} -b ${batch} --fix_seed --log_path ${logs} -lr ${lr} --weight_decay ${w};
            python main_more.py -m ${m} -lr ${lr} --data ${dataset} --sampler 0 -s $n -e ${epoch} -b ${batch} --fix_seed --weighted --log_path ${logs} --weight_decay ${w};
            python main_more.py -m ${m} -lr ${lr} --data ${dataset} --sampler 3 -s $n -e ${epoch} -b ${batch} --fix_seed --weighted --log_path ${logs} --weight_decay ${w}
        done
    done
done

#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"


dataset='gowalla'
lr=0.01
epoch=300
batch=4096
weight_decay=0.001
anneal=0.001
main_file='main.py'
logs='log_no_ts_go'


# python ${main_file} -lr ${lr} --data ${dataset} --sampler 0 -s 5 -e ${epoch} -b ${batch} --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

for n in 5
do
    # # BPR
    # python ${main_file} -lr ${lr} --data ${dataset} --sampler 0 -s 5 -e ${epoch} -b ${batch} --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

    # # IS-Uniform
    # python ${main_file} -lr ${lr} --data ${dataset} --sampler 0 -s 5 -e ${epoch} -b ${batch} --weighted --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};



    # python ${main_file} -lr ${lr} --data ${dataset} --sampler 1 --pool_size 200 --sample_size 200 -s $n -e ${epoch} -b ${batch} --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

    # random sample + wi
    # python ${main_file} -lr ${lr} --data ${dataset} --sampler 2 --pool_size 200 --sample_size 200 -s $n -e ${epoch} -b ${batch} --weighted --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

    # IS-Pop
    python ${main_file} -lr ${lr} --data ${dataset} --sampler 3 --pool_size 200 --sample_size 200 -s $n -e ${epoch} -b ${batch} --weighted  --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

    # Two - pass pop
    python ${main_file} -lr ${lr} --data ${dataset} --sampler 4 --pool_size 200 --sample_size 200 -s $n -e ${epoch} -b ${batch} --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

    python ${main_file} -lr ${lr} --data ${dataset} --sampler 5 --pool_size 200 --sample_size 200 -s $n -e ${epoch} -b ${batch} --weighted --fix_seed --log_path ${logs} --weight_decay ${weight_decay} --anneal ${anneal};

done
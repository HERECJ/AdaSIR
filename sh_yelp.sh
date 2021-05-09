#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"
dataset='yelp'
logs='logs_yelp'

# for n in 3 5 10
# do
#     for s in 0 1 3
#     # for s in 3
#     do
#         echo "sampler No.${s}";  
#         python main.py -lr 0.001 --data ${dataset} --pool_size 200 --sample_size 200 --sampler $s -s $n -e 500 -b 4096 --log_path 'logs_go_more_33' --step_size 20 --weighted True;
#         python main.py -lr 0.001 --data ${dataset} --pool_size 200 --sample_size 200 --sampler $s -s $n -e 500 -b 4096 --log_path 'logs_go_more_33' --step_size 20;
#     done
# done

for n in 5
do
    # # BPR
    python main.py -lr 0.001 --data ${dataset} --sampler 0 -s $n -e 500 -b 4096 --log_path ${logs};

    # # IS-Uniform
    python main.py -lr 0.001 --data ${dataset} --sampler 0 -s $n -e 500 -b 4096 --weighted --log_path ${logs};

    # Two - Pass
    # random sample + random weight
    python main.py -lr 0.001 --data ${dataset} --sampler 1 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path ${logs};

    # random sample + softmax weight
    python main.py -lr 0.001 --data ${dataset} --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --random_flag --log_path ${logs};

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data ${dataset} --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path ${logs};

    # Q:pop

    # random sample + softmax weight
    python main.py -lr 0.001 --data ${dataset} --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --random_flag --log_path ${logs};

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data ${dataset} --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path ${logs};
done

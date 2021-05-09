#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"

# for b in 3 5 10
# # for b in 1 2
# do
#     for s in 0 1 3
#     do
#         echo "sampler No.${s}";  
#         python main.py -lr 0.001 --data ml10M --pool_size 200 --sample_size 200 --sampler $s -s $b -e 500 -b 4096 --log_path 'logs_10Ms_more_33' --weighted True;
#         python main.py -lr 0.001 --data ml10M --pool_size 200 --sample_size 200 --sampler $s -s $b -e 500 -b 4096 --log_path 'logs_10Ms_more_33';
#     done
# done

for n in 5
do
    # BPR
    python main.py -lr 0.001 --data ml10M --sampler 0 -s $n -e 500 -b 4096 --log_path 'logs_ml10M';

    # IS-Uniform
    python main.py -lr 0.001 --data ml10M --sampler 0 -s $n -e 500 -b 4096 --weighted --log_path 'logs_ml10M';



    # Two - Pass
    # random sample + random weight
    python main.py -lr 0.001 --data ml10M --sampler 1 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path 'logs_ml10M'

    # random sample + softmax weight
    python main.py -lr 0.001 --data ml10M --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --random_flag --log_path 'logs_ml10M'

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data ml10M --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path 'logs_ml10M'

    # Q:pop

    # random sample + softmax weight
    python main.py -lr 0.001 --data ml10M --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --random_flag --log_path 'logs_ml10M'

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data ml10M --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path 'logs_ml10M'
done

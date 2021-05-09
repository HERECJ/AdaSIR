#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"

# for n in 3 5 10
# do
#     for s in 0 1 3
#     # for s in 3
#     do
#         echo "sampler No.${s}";  
#         python main.py -lr 0.001 --data gowalla --pool_size 200 --sample_size 200 --sampler $s -s $n -e 500 -b 4096 --log_path 'logs_go_more_33' --step_size 20 --weighted True;
#         python main.py -lr 0.001 --data gowalla --pool_size 200 --sample_size 200 --sampler $s -s $n -e 500 -b 4096 --log_path 'logs_go_more_33' --step_size 20;
#     done
# done
# python main.py -lr 0.001 --data gowalla --sampler 4 -s 5 -e 500 -b 4096 --weighted --log_path 'logs_gowalla';
for n in 5
do
    # # BPR
    python main.py -lr 0.001 --data gowalla --sampler 0 -s $n -e 500 -b 4096 --log_path 'logs_gowalla';

    # # IS-Uniform
    python main.py -lr 0.001 --data gowalla --sampler 0 -s $n -e 500 -b 4096 --weighted --log_path 'logs_gowalla';

    # # Two - Pass
    # # random sample + random weight
    # python main.py -lr 0.001 --data gowalla --sampler 1 -s $n -e 500 -b 4096 --weighted --log_path 'logs_go'

    # # random sample + softmax weight
    # python main.py -lr 0.001 --data gowalla --sampler 2 -s $n -e 500 -b 4096 --weighted --random_flag --log_path 'logs_go'

    # # multinomial sample + softmax weight
    # python main.py -lr 0.001 --data gowalla --sampler 2 -s $n -e 500 -b 4096 --weighted --log_path 'logs_go'

    # # Q:pop

    # # random sample + softmax weight
    # python main.py -lr 0.001 --data gowalla --sampler 3 -s $n -e 500 -b 4096 --weighted --random_flag --log_path 'logs_go'

    # # multinomial sample + softmax weight
    # python main.py -lr 0.001 --data gowalla --sampler 3 -s $n -e 500 -b 4096 --weighted --log_path 'logs_go'

    # Two - Pass
    # random sample + random weight
    python main.py -lr 0.001 --data gowalla --sampler 1 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path 'logs_gowalla'

    # random sample + softmax weight
    python main.py -lr 0.001 --data gowalla --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --random_flag --log_path 'logs_gowalla'

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data gowalla --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path 'logs_gowalla'

    # Q:pop

    # random sample + softmax weight
    python main.py -lr 0.001 --data gowalla --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --random_flag --log_path 'logs_gowalla'

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data gowalla --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 500 -b 4096 --weighted --log_path 'logs_gowalla'
done

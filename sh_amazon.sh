#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"

# for n in 3 5 10
# do
#     # for s in 0 1 3
#     for s in 1 3
    
#     do
#         echo "sampler No.${s}";  
#         python main.py -lr 0.001 --data amazoni --pool_size 200 --sample_size 200 --sampler $s -s $n -e 800 -b 8192 --num_workers 12 --log_path 'logs_amazoni' --step_size 20 --weighted True;
#         python main.py -lr 0.001 --data amazoni --pool_size 200 --sample_size 200 --sampler $s -s $n -e 800 -b 8192 --num_workers 12 --log_path 'logs_amazoni' --step_size 20;
#     done
# done

for n in 5
do
    # BPR
    python main.py -lr 0.001 --data amazoni --sampler 0 -s $n -e 800 -b 8192 --log_path 'logs_amazoni' --step_size 30 --num_workers 12;

    # IS-Uniform
    python main.py -lr 0.001 --data amazoni --sampler 0 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni' --step_size 30 --num_workers 12;


    # Two - Pass
    # random sample + random weight
    # python main.py -lr 0.001 --data amazoni --sampler 1 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni'

    # # random sample + softmax weight
    # python main.py -lr 0.001 --data amazoni --sampler 2 -s $n -e 800 -b 8192 --weighted --random_flag --log_path 'logs_amazoni'

    # # multinomial sample + softmax weight
    # python main.py -lr 0.001 --data amazoni --sampler 2 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni'

    # # Q:pop

    # # random sample + softmax weight
    # python main.py -lr 0.001 --data amazoni --sampler 3 -s $n -e 800 -b 8192 --weighted --random_flag --log_path 'logs_amazoni'

    # # multinomial sample + softmax weight
    # python main.py -lr 0.001 --data amazoni --sampler 3 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni'




    # Two - Pass
    # random sample + random weight
    python main.py -lr 0.001 --data amazoni --sampler 1 --pool_size 200 --sample_size 200 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni' --num_workers 12 --step_size 20

    # random sample + softmax weight
    python main.py -lr 0.001 --data amazoni --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 800 -b 8192 --weighted --random_flag --log_path 'logs_amazoni' --num_workers 12 --step_size 20

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data amazoni --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni' --num_workers 12 --step_size 20

    # Q:pop

    # random sample + softmax weight
    python main.py -lr 0.001 --data amazoni --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 800 -b 8192 --weighted --random_flag --log_path 'logs_amazoni' --num_workers 12 --step_size 20

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --data amazoni --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 800 -b 8192 --weighted --log_path 'logs_amazoni' --num_workers 12 --step_size 20
 
done


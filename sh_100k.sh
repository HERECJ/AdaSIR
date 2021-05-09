#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"
# python eval_std.py -lr 0.001 --sampler 0 -s 5 -e 300 -b 128
for n in 5
do
    # BPR
    python main.py -lr 0.001 --sampler 0 -s $n -e 300 -b 128 --log_path 'logs_ml100k';

    # IS-Uniform
    python main.py -lr 0.001 --sampler 0 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k';

    # Two - Pass
    # random sample + random weight
    # python main.py -lr 0.001 --sampler 1 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k'

    # # random sample + softmax weight
    # python main.py -lr 0.001 --sampler 2 -s $n -e 300 -b 128 --weighted --random_flag --log_path 'logs_ml100k'

    # # multinomial sample + softmax weight
    # python main.py -lr 0.001 --sampler 2 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k'

    # # Q:pop

    # # random sample + softmax weight
    # python main.py -lr 0.001 --sampler 3 -s $n -e 300 -b 128 --weighted --random_flag --log_path 'logs_ml100k'

    # # multinomial sample + softmax weight
    # python main.py -lr 0.001 --sampler 3 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k'




    # Two - Pass
    # random sample + random weight
    python main.py -lr 0.001 --sampler 1 --pool_size 200 --sample_size 200 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k'

    # random sample + softmax weight
    python main.py -lr 0.001 --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 300 -b 128 --weighted --random_flag --log_path 'logs_ml100k'

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --sampler 2 --pool_size 200 --sample_size 200 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k'

    # Q:pop

    # random sample + softmax weight
    python main.py -lr 0.001 --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 300 -b 128 --weighted --random_flag --log_path 'logs_ml100k'

    # multinomial sample + softmax weight
    python main.py -lr 0.001 --sampler 3 --pool_size 200 --sample_size 200 -s $n -e 300 -b 128 --weighted --log_path 'logs_ml100k'
done

python main.py -lr 0.001 --sampler 4 --pool_size 200 --sample_size 200 -s 5 -e 300 -b 128 --weighted --log_path 'logs_ml100k'
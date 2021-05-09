#!/bin/bash

export CUDA_VISIBLE_DEVICES="6"

for n in 3
do
    for s in 3
    do
        echo "sampler No.${s}, batch_size : 128, sample_num : ${n}"; 
        python main.py -lr 0.001 --pool_size 200 --sampler $s -s $n -e 300 -b 128 --log_path 'logs_test' --weighted True;
    done
done

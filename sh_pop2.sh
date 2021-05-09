#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"
python main.py -lr 0.001 --data gowalla --sampler 4 -s 5 -e 500 -b 4096 --weighted --log_path 'logs_gowalla';
python main.py -lr 0.001 --data amazoni --sampler 4 -s 5 -e 800 -b 8192 --weighted --log_path 'logs_amazoni' --step_size 30 --num_workers 12;

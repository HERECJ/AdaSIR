#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"

python main.py -lr 0.001 --sampler 4 --pool_size 200 --sample_size 200 -s 5 -e 300 -b 128 --weighted --log_path 'logs_ml100k';
python main.py -lr 0.001 --data ml10M --sampler 4 --pool_size 200 --sample_size 200 -s 5 -e 500 -b 4096 --weighted --log_path 'logs_ml10M';
python main.py -lr 0.001 --data yelp --sampler 4 --pool_size 200 --sample_size 200 -s 5 -e 500 -b 4096 --weighted --log_path 'logs_yelp';
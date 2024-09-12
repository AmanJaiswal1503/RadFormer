#!/bin/bash

python train.py \
        --img_dir="data/imgs" \
        --train_list="data/train.txt" \
        --val_list="data/test.txt" \
        --save_name="radformer" \
        --lr=0.003 \
        --optim="sgd" \
        --batch_size=32 \
        --global_weight=0.6 \
        --fusion_weight=0.1 \
        --local_weight=0.3 \
        --epochs=100 \
        --pretrain \
        --load_local \
        --num_layers=4 \
        --wandb_project="radformer" \
        --wandb_name="replication"

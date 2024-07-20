#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

# 指定GPU分布式训练：CUDA_VISIBLE_DEVICES=0,1,3,4 ./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all_mtsd-traffic-sign.py 4 --work-dir work_dir_MMGD78_finetune_mtsd
#!/bin/bash

#cd ../..

# custom config
DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
TRAINER=ZeroshotCLIP

CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
for DATASET in imagenet_1k 
do
    CUDA_VISIBLE_DEVICES=4 python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/${DATASET} \
    --eval-only
done
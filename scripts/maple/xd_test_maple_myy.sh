#!/bin/bash

#cd ../..

# custom config
DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
TRAINER=Calibration
EXP_NAME=L1_match_classwise

SEED=2
# ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch
CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets_calib
SHOTS=16

for DATASET in imagenet_r imagenet_a imagenet_sketch imagenetv2 imagenet_1k
do
DIR=output/evaluation/${TRAINER}/${EXP_NAME}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=4 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output_run/imagenet_1k/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 2 \
    --eval-only
fi
done
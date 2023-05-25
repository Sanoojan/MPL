#!/bin/bash
#SBATCH --job-name=cocoop-zero_shot
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=default-short

#cd ../..

# custom config
DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
TRAINER=CoCoOp

DATASET=imagenet_1k
SEED=2

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16
DEVICE=12

SHOTS=16
LOADEP=2

# DIR=output/cocoop/xtrain_text_to_text/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
DIR=output/cocoop/xtrain/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
# if [ -d "$DIR" ]; then
#     echo "Results are available in ${DIR}. Resuming..."
#     CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#     --output-dir ${DIR} \
#     DATASET.NUM_SHOTS ${SHOTS}
# else
#     echo "Run this job and save the output to ${DIR}"

#     CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#     --output-dir ${DIR} \
#     DATASET.NUM_SHOTS ${SHOTS}
# fi

CFG=vit_b16_c4_ep10_batch1_ctxv1_text_to_text

for DATASET in imagenet_r imagenet_a imagenet_sketch imagenetv2 imagenet_1k
do

DIR_SAVE=${DIR}/evaluation/${DATASET}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_SAVE} \
    --model-dir ${DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} 

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_SAVE} \
    --model-dir ${DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} 
fi
done
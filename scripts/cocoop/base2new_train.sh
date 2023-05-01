#!/bin/bash
#SBATCH --job-name=cocoop
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=default-long

#cd ../..

# custom config
DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
TRAINER=CoCoOp

DATASET=imagenet_1k
SEED=2

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16

# DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
# TRAINER=CoOp
# EXP_NAME=Baseline
# DATASET=imagenet_1k
# SEED=2

# CFG=vit_b16_ep50
# SHOTS=16


# CTP=end  # class token position (end or middle)
# NCTX=16 # number of context tokens
# SHOTS=16  # number of shots (1, 2, 4, 8, 16)
# CSC=True  # class-specific context (False or True)


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi


LOADEP=10
SUB=new


for DATASET in imagenet_r imagenet_a imagenet_sketch imagenetv2
do

DIR_SAVE=${DIR}/evaluation/${DATASET}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_SAVE} \
    --model-dir ${DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi

done
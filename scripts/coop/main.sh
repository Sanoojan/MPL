#!/bin/bash
#SBATCH --job-name=coop
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=default-long

#cd ../..

cd /nfs/users/ext_sanoojan.baliah/Sanoojan/MPL
source ~/.bashrc
conda activate mpl2

# custom config
DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
TRAINER=CoOp
EXP_NAME=Baseline
DATASET=imagenet_1k
SEED=2

CFG=vit_b16_ep50
SHOTS=16


CTP=end  # class token position (end or middle)
NCTX=16 # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=True  # class-specific context (False or True)

for SEED in 2
do
    DIR=output_coop/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

DIR=output_coop/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
for DATASET in imagenet_r imagenet_a imagenet_sketch imagenetv2
do
DIR_SAVE=${DIR}/evaluation/${DATASET}
for SEED in 2
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir ${DIR} \
    --output-dir ${DIR_SAVE} \
    --load-epoch 50 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done
done
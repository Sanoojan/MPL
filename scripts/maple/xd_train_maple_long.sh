#!/bin/bash
#SBATCH --job-name=maple_baseline
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=default-short

#cd ../..

cd /nfs/users/ext_sanoojan.baliah/Sanoojan/MPL
source ~/.bashrc
conda activate mpl2

# custom config
DATA="/nfs/users/ext_sanoojan.baliah/Sanoojan/data"
TRAINER=MaPLe
EXP_NAME=Baseline
DATASET=imagenet_1k
SEED=2

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16


DIR=output/${EXP_NAME}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} > run_outs/${EXP_NAME}_${DATASET}_${TRAINER}_${CFG}_${SHOTS}shots_seed${SEED}.out
fi

for DATASET in imagenet_r imagenet_a imagenet_sketch imagenetv2 imagenet_1k
do
DIR_SAVE=${DIR}/evaluation/${DATASET}
if [ -d "$DIR_SAVE" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_SAVE} \
    --model-dir ${DIR} \
    --load-epoch 2
    
    
    
    
    
     \
    --eval-only
fi
done


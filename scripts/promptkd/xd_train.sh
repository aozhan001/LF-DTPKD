#!/bin/bash

# custom config
DATA='/data2/workspace_hyw/promptkd/promptkd_data'
TRAINER=PromptKD

DATASET=$1 # 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101' 'caltech101'
SEED=$2
GPU_ID=$3

CFG=vit_b16_c2_ep20_batch8_4+4ctx_cross_datasets
SHOTS=0

DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed_${SEED}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT 1000.0 \
    TRAINER.MODAL cross\

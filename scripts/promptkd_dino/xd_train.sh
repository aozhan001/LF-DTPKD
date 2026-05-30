#!/bin/bash

DATA="/data2/workspace_hyw/promptkd/promptkd_data"
TRAINER=PromptKDDINO
CFG=vit_b16_dinov2_b14_cross_datasets
SHOTS=0

DATASET=$1
SEED=$2
GPU_ID=$3
KD_WEIGHT=${KD_WEIGHT:-1.0}
DINO_WEIGHT=${DINO_WEIGHT:-2.0}
DINO_CKPT=${DINO_CKPT:-"./teacher_model/ImageNet-xd/DINOv2Teacher/dinov2_vitb14.pth"}
DINO_REPO_OR_DIR=${DINO_REPO_OR_DIR:-"/data2/workspace_hyw/promptkd/LF-DTPKD/dinov2"}

DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed_${SEED}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/PromptKDDINO/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.PROMPTKD_DINO.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD_DINO.KD_WEIGHT ${KD_WEIGHT} \
    TRAINER.PROMPTKD_DINO.DINO_WEIGHT ${DINO_WEIGHT} \
    TRAINER.PROMPTKD_DINO.DINO_CKPT "${DINO_CKPT}" \
    TRAINER.PROMPTKD_DINO.DINO_REPO_OR_DIR "${DINO_REPO_OR_DIR}" \
    TRAINER.PROMPTKD_DINO.USE_CLIP_TEACHER_CKPT False \
    TRAINER.MODAL cross

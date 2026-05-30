#!/bin/bash

DATA="/data2/workspace_hyw/promptkd/promptkd_data"
TRAINER=DINOv2Pretrain
CFG=vitb14_16shot
SHOTS=16

DATASET=$1
SEED=$2
GPU_ID=$3
SAVE_DIR=${SAVE_DIR:-"./teacher_model/ImageNet-xd/DINOv2Teacher"}
SAVE_NAME=${SAVE_NAME:-"dinov2_vitb14.pth"}
INIT_CKPT=${INIT_CKPT:-"./clip/dinov2_vitb14_pretrain.pth"}
DINO_REPO_OR_DIR=${DINO_REPO_OR_DIR:-"/data2/workspace_hyw/promptkd/LF-DTPKD/dinov2"}

DIR=output/dinov2_pretrain/xd/${DATASET}/shots_${SHOTS}/${CFG}/seed_${SEED}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/DINOv2Pretrain/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.SUBSAMPLE_CLASSES all \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.DINOV2_PRETRAIN.SAVE_DIR "${SAVE_DIR}" \
    TRAINER.DINOV2_PRETRAIN.SAVE_NAME "${SAVE_NAME}" \
    TRAINER.DINOV2_PRETRAIN.INIT_CKPT "${INIT_CKPT}" \
    TRAINER.DINOV2_PRETRAIN.DINO_REPO_OR_DIR "${DINO_REPO_OR_DIR}"

#!/bin/bash
TRAIN_DATASET_PATH="./downloads/TinyStories_dev_data_tokenized.npy"
VAL_DATASET_PATH="./downloads/TinyStories_dev_data_tokenized.npy"

# Run the training script with appropriate arguments
uv run cs336_basics/train.py \
  --training-dataset "${TRAIN_DATASET_PATH}" \
  --validation-dataset "${VAL_DATASET_PATH}" \
  --batch-size 128 \
  --context-length 256 \
  --vocab-size 10000 \
  --num-layers 2 \
  --d-model 256 \
  --num-heads 2 \
  --d-ff 640 \
  --rope-theta 10000.0 \
  --num-iterations 1000 \
  --log-every 1 \
  --eval-every 100 \
  --save-every 200\
  --device "cuda" \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --weight-decay 0.01 \
  --beta1 0.9 \
  --beta2 0.999 \
  --warmup-iters 10 \
  --max-grad-norm 1.0 \
  --checkpoint-dir "./downloads/checkpoint/" \
  --wandb-project "cs336-transformer-lm" \
  --wandb-name "training-test-run-2" \
  --verbose
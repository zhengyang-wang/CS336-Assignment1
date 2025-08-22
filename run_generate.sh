#!/bin/bash

# Define paths for the checkpoint and tokenizer files
CHECKPOINT_PATH="./downloads/checkpoint/training-test-run-2/checkpoint_001000.pt"
TOKENIZER_VOCAB_PATH="./train-bpe-tinystories-vocab.json"
TOKENIZER_MERGES_PATH="./train-bpe-tinystories-merges.txt"

# Run the generation script with appropriate arguments
uv run cs336_basics/generate.py \
  --tokenizer-vocab-path "${TOKENIZER_VOCAB_PATH}" \
  --tokenizer-merges-path "${TOKENIZER_MERGES_PATH}" \
  --tokenizer-eos-tokens "<|endoftext|>" \
  --context-length 256 \
  --vocab-size 10000 \
  --num-layers 2 \
  --d-model 256 \
  --num-heads 2 \
  --d-ff 640 \
  --rope-theta 10000.0 \
  --device "cuda" \
  --load-from "${CHECKPOINT_PATH}" \
  --max-gen-tokens 128 \
  --temperature 0.5 \
  --top-p 0.9
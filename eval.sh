#!/bin/bash
DEVICE=$(python3 src/constants.py)

lm-eval run --model hf \
    --model_args pretrained=model_weights/qwen-adamw \
    --tasks piqa \
    --output_path metrics/qwen-adamw-res \
    --device $DEVICE

lm-eval run --model hf \
    --model_args pretrained=model_weights/qwen-muon \
    --tasks piqa \
    --output_path metrics/qwen-muon-res \
    --device $DEVICE

lm-eval run --model hf \
    --model_args pretrained=model_weights/qwen-muon-adamw \
    --tasks piqa \
    --output_path metrics/qwen-muon-adamw-res \
    --device $DEVICE

#!/bin/bash

# 参数解析
while getopts "m:g:" opt; do
  case $opt in
    m) MODEL=$OPTARG ;;
    g) GPU_ID=$OPTARG ;;
    *) echo "Usage: $0 -m [transformer|tcn|lstm] -g [gpu_id]" >&2
       exit 1 ;;
  esac
done

# 默认参数
MODEL=${MODEL:-transformer}
GPU_ID=${GPU_ID:-0}

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 训练函数
train() {
    case $MODEL in
        transformer)
            echo "Training Transformer model on GPU $GPU_ID..."
            python -m transformer_trainer.train_transformer
            ;;
        tcn)
            echo "Training TCN model on GPU $GPU_ID..."
            python -m transformer_trainer.train_tcn
            ;;
        lstm)
            echo "Training LSTM model on GPU $GPU_ID..."
            python -m transformer_trainer.train_lstm
            ;;
        *)
            echo "Invalid model type: $MODEL"
            exit 1
            ;;
    esac
}

# 执行训练
train
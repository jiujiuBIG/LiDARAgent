#!/bin/bash

# 模型路径配置
BASE_MODEL="/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/LLMs/code/LLM_Finetune/model/DeepSeek-R1-Distill-Qwen-32B"
ADAPTER_PATH="/mnt/Huxiaomei2/May/Huxiaomei/mcp_llm_new/outputs/auto_lora_sft_1219/checkpoint-100"
CHAT_TEMPLATE="/mnt/Huxiaomei2/May/Huxiaomei/mcp_llm_new/outputs/auto_lora_sft_1219/checkpoint-100/chat_template.jinja"

# 服务配置
HOST="0.0.0.0"
PORT="8000"
MODEL_NAME="forestmind"

# GPU 配置：使用 4 张 3090
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=========================================="
echo "启动 vLLM + LoRA 服务器"
echo "=========================================="
echo "基础模型: $BASE_MODEL"
echo "LoRA Adapter: $ADAPTER_PATH"
echo "监听地址: $HOST:$PORT"
echo "服务名称: $MODEL_NAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --lora-modules "forestmind=$ADAPTER_PATH" \
    --tensor-parallel-size 4 \
    --dtype half \
    --max-model-len 32000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --enable-lora \
    --host "$HOST" \
    --port "$PORT" \
    --chat-template "$CHAT_TEMPLATE"

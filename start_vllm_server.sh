#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# vLLM OpenAI API server startup script
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults can be overridden by environment variables, for example:
# MODEL_PATH=/path/to/base/model LORA_PATH=/path/to/lora ./start_vllm_server.sh
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/llm_model/DeepSeek-R1-Distill-Qwen-32B}"
LORA_NAME="${LORA_NAME:-forestmind}"
LORA_PATH="${LORA_PATH:-${SCRIPT_DIR}/llm_model/checkpoints}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${LORA_PATH}/chat_template.jinja}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DTYPE="${DTYPE:-half}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Use the first 4 GPUs by default. Override with CUDA_VISIBLE_DEVICES=0,1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/vllm_server_${TIMESTAMP}.log"

die() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARN: $*" >&2
}

check_file() {
    local path="$1"
    local desc="$2"

    [[ -f "${path}" ]] || die "${desc} not found: ${path}"
}

check_dir() {
    local path="$1"
    local desc="$2"

    [[ -d "${path}" ]] || die "${desc} not found: ${path}"
}

echo "============================================================"
echo "Checking runtime environment"
echo "============================================================"

command -v python >/dev/null 2>&1 || die "python was not found. Please activate the correct Conda/venv environment."

echo "Python path: $(command -v python)"
echo "Python version: $(python --version 2>&1)"

if ! python -c "import vllm" >/dev/null 2>&1; then
    die "vLLM is not installed in the current Python environment."
fi

echo "vLLM path: $(python -c 'import vllm; print(vllm.__file__)')"

echo
echo "============================================================"
echo "Checking base model"
echo "============================================================"

check_dir "${MODEL_PATH}" "Base model directory"
check_file "${MODEL_PATH}/config.json" "Base model config.json"

if [[ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    warn "model.safetensors.index.json was not found. Please verify the base model files are complete."
fi

MODEL_WEIGHT_COUNT="$(
    find "${MODEL_PATH}" -maxdepth 1 -type f -name "*.safetensors" | wc -l
)"

echo "Detected model weight files: ${MODEL_WEIGHT_COUNT}"

if [[ "${MODEL_WEIGHT_COUNT}" -eq 0 ]]; then
    die "no .safetensors weight files found in base model directory: ${MODEL_PATH}"
fi

echo
echo "============================================================"
echo "Checking LoRA adapter"
echo "============================================================"

check_dir "${LORA_PATH}" "LoRA directory"
check_file "${LORA_PATH}/adapter_config.json" "LoRA adapter_config.json"

if [[ ! -f "${LORA_PATH}/adapter_model.safetensors" ]] &&
   [[ ! -f "${LORA_PATH}/adapter_model.bin" ]]; then
    die "LoRA weight file not found. Expected ${LORA_PATH}/adapter_model.safetensors or ${LORA_PATH}/adapter_model.bin"
fi

check_file "${CHAT_TEMPLATE}" "Chat template"

echo
echo "============================================================"
echo "Checking GPU"
echo "============================================================"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found. Please check NVIDIA driver and CUDA setup."

nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader

VISIBLE_GPU_COUNT="$(
    tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" |
    sed '/^[[:space:]]*$/d' |
    wc -l
)"

if [[ "${VISIBLE_GPU_COUNT}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
    die "CUDA_VISIBLE_DEVICES has ${VISIBLE_GPU_COUNT} GPU(s), but tensor-parallel-size is ${TENSOR_PARALLEL_SIZE}. Please keep them consistent."
fi

echo
echo "============================================================"
echo "Checking port"
echo "============================================================"

if command -v ss >/dev/null 2>&1; then
    if ss -lnt | awk '{print $4}' | grep -qE ":${PORT}$"; then
        ss -lntp | grep ":${PORT}" || true
        die "port ${PORT} is already in use. Stop the old service or change PORT."
    fi
else
    warn "ss command was not found. Skipping port check."
fi

echo
echo "============================================================"
echo "Starting vLLM service"
echo "============================================================"
echo "Base model: ${MODEL_PATH}"
echo "LoRA name: ${LORA_NAME}"
echo "LoRA path: ${LORA_PATH}"
echo "Chat template: ${CHAT_TEMPLATE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Tensor parallel: ${TENSOR_PARALLEL_SIZE}"
echo "dtype: ${DTYPE}"
echo "Max model length: ${MAX_MODEL_LEN}"
echo "Max sequences: ${MAX_NUM_SEQS}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Listen URL: http://${HOST}:${PORT}"
echo "Log file: ${LOG_FILE}"
echo "============================================================"
echo
echo "vLLM will keep running in the foreground. Press Ctrl+C to stop it."
echo

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --lora-modules "${LORA_NAME}=${LORA_PATH}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-lora \
    --host "${HOST}" \
    --port "${PORT}" \
    --chat-template "${CHAT_TEMPLATE}" \
    2>&1 | tee "${LOG_FILE}"

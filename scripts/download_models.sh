#!/usr/bin/env bash
# Pre-download models to the shared `models` volume so vLLM startup doesn't pull
# multi-GB weights on first invocation.
set -euo pipefail

: "${HUGGING_FACE_HUB_TOKEN:?HUGGING_FACE_HUB_TOKEN must be set (export it before running)}"
: "${VLLM_LLM_MODEL:=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}"
: "${VLLM_OCR_MODEL:=datalab-to/chandra-ocr-2}"
: "${EMBEDDING_MODEL:=BAAI/bge-m3}"

VOLUME_NAME="${MODELS_VOLUME:-llm-engine_models}"
HF_HOME_INNER="/models/hf"

docker run --rm \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  -e HF_HOME="${HF_HOME_INNER}" \
  -v "${VOLUME_NAME}:/models" \
  --entrypoint bash \
  python:3.11-slim \
  -c "pip install --no-cache-dir huggingface_hub && \
      huggingface-cli download ${VLLM_LLM_MODEL} --local-dir-use-symlinks False && \
      huggingface-cli download ${VLLM_OCR_MODEL}  --local-dir-use-symlinks False && \
      huggingface-cli download ${EMBEDDING_MODEL} --local-dir-use-symlinks False"

echo "Models cached to volume: ${VOLUME_NAME}"

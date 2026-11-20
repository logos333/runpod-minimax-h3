#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS_ROOT="${MODELS_ROOT:-/runpod-volume/models}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"
AUTO_SYNC_ON_BOOT="${AUTO_SYNC_ON_BOOT:-true}"

echo "[boot] MiniMax-H3 Serverless Worker starting..."
echo "[boot] Models root: ${MODELS_ROOT}"

mkdir -p "${MODELS_ROOT}/diffusion_models" \
         "${MODELS_ROOT}/text_encoders" \
         "${MODELS_ROOT}/vae" \
         "${MODELS_ROOT}/loras" \
         /app/ComfyUI/input \
         /app/ComfyUI/output

if [ "${AUTO_SYNC_ON_BOOT}" = "true" ]; then
  echo "[boot] Checking/syncing required MiniMax-H3 weights..."
  python3 -m worker.model_sync
fi

echo "[boot] Starting headless ComfyUI on ${COMFY_HOST}:${COMFY_PORT}..."
python3 /app/ComfyUI/main.py \
  --listen "${COMFY_HOST}" \
  --port "${COMFY_PORT}" \
  --extra-model-paths-config /app/docker/extra_model_paths.yaml \
  --disable-auto-launch \
  --disable-metadata \
  > /tmp/comfyui.log 2>&1 &

COMFY_PID=$!

# Wait until ComfyUI HTTP server is responsive
for i in $(seq 1 120); do
  if curl -sf "http://${COMFY_HOST}:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
    echo "[boot] ComfyUI ready (pid=${COMFY_PID}) after ${i}s"
    break
  fi
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "[boot] ERROR: ComfyUI exited prematurely. Log:"
    cat /tmp/comfyui.log
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://${COMFY_HOST}:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
  echo "[boot] ERROR: ComfyUI did not become ready within 120s. Log:"
  tail -n 100 /tmp/comfyui.log
  exit 1
fi

echo "[boot] Starting RunPod Serverless handler..."
exec python3 -u -m worker.handler

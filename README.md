# MiniMax-H3 on RunPod Serverless

Headless **RunPod Serverless** worker + parallel **multi-scene orchestrator** running **MATLOWAI MiniMax-H3 Fused Turbo (`int8_convrot`)** with native synchronized stereo audio, multi-reference conditioning (`Ref2VA`), first/last-frame hybrid anchoring (`MiniMaxH3AddGuide`), SLA block-sparse attention, and `ComfyUI-MiniMax-H3-LongMedia`.

## Architecture

- **Worker image:** `ghcr.io/logos333/minimax-h3-worker:latest` (PyTorch `cu130` + ComfyUI + `H3SLAAttention` + `ComfyUI-MiniMax-H3-LongMedia` + `ComfyUI-KJNodes` + `ComfyUI-MAINodes` + FFmpeg).
- **Models (~37.3 GB on a 60 GB RunPod Network Volume):**
  - `MATLOWAI/minimax-h3-fused-turbo-int8-convrot` (`19.54 GB` — single file for both `Ref2VA` & `FL2VA`, 4-step Turbo baked in)
  - `Comfy-Org/MiniMax-H3` `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (`14.61 GB`)
  - `Comfy-Org/MiniMax-H3` `minimax_h3_video_vae_int8_convrot.safetensors` (`2.62 GB`)
  - `Comfy-Org/MiniMax-H3` `minimax_h3_audio_vae_fp32.safetensors` (`0.56 GB`)
- **Multi-scene (2–3 min) flow:** Independent `ref2va` scenes fan out in parallel across RunPod Serverless workers; scenes with `"continue_from_previous": true` chain via `hybrid` mode (`ref_images` + previous scene's `last_frame`); FFmpeg stitches clips with smooth audio crossfade.

## Quickstart

### 1. Provision RunPod (after topping up balance)
```bash
uv run python -m scripts.deploy_runpod
```
Creates the 60 GB Network Volume, Serverless Template, and Serverless Endpoint, and writes `RUNPOD_ENDPOINT_ID` into `.env`.

### 2. First-time Weight Sync (~2–4 min once)
```bash
uv run minimax-h3 warmup
```
Downloads only the 4 exact files (~37.3 GB) onto `/runpod-volume/models`. Subsequent workers boot warm in seconds.

### 3. Generate a Single Multi-Reference Clip
```bash
uv run minimax-h3 clip \
  --ref-image ./me.jpg \
  --ref-image ./dress.jpg \
  --ref-image ./bg.jpg \
  --duration 10 \
  --prompt "subject_definitions: <Subject 1> is the person in <Picture 1> wearing <Picture 2> in <Picture 3>..."
```

### 4. Generate a Multi-Scene Video (or Re-roll 1 Scene)
```bash
uv run minimax-h3 storyboard examples/multi_scene_storyboard.json
# Re-render only one scene and re-stitch:
uv run minimax-h3 storyboard examples/multi_scene_storyboard.json --only-scene scene_02_continuous_walk
```

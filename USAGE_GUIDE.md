# MiniMax-H3 on RunPod Serverless — Complete Usage & Operations Guide

This guide covers everything you need to **start, stop, monitor, call, and completely tear down** your MiniMax-H3 video + synchronized audio service on RunPod Serverless.

---

## 1. What Is Deployed & How Billing Works

| Resource | ID / Name | Cost |
|---|---|---|
| **Worker Docker Image** | `ghcr.io/logos333/minimax-h3-worker:latest` (public, auto-built via GitHub Actions) | **\$0** |
| **Serverless Template** | `minimax-h3-serverless-template` (`gmhkcxi0nn`) | **\$0** |
| **Network Volume (60 GB, `EU-RO-1`)** | `minimax-h3-models` (`hdhyqoz7jb`) — holds ~37.3 GB weights | **~\$4.20/month** (~\$0.0058/hr, billed while volume exists) |
| **Serverless Endpoint** | `minimax-h3-serverless` (`iksf1j15rktqtl`) | **\$0 when idle** (`workersMin=0`, shuts down 10s after job finishes); pay-per-second only while generating |

**GPU Pools configured (in priority order, `EU-RO-1`):** `ADA_32_PRO` (RTX 5090 32GB), `AMPERE_80` (A100 80GB), `BLACKWELL_96` (RTX PRO 6000 Blackwell 96GB), `ADA_48_PRO` (L40S / RTX 6000 Ada 48GB), `ADA_24` (RTX 4090 24GB fallback).

**Model stack stored on `/runpod-volume/models` (~37.3 GB total):**
- `diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` (19.54 GB — **single fused model** for both `Ref2VA` + `FL2VA` with 4-step Turbo baked in)
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (14.61 GB)
- `vae/minimax_h3_video_vae_int8_convrot.safetensors` (2.62 GB)
- `vae/minimax_h3_audio_vae_fp32.safetensors` (0.56 GB)

---

## 2. Check Status, Start, Stop & Complete Teardown

All lifecycle commands run from `/Users/temurkhoshimov/myData/projects/runpod.io`.

### Check balance, active spend, and live worker health
```bash
uv run python -m scripts.manage_runpod status
```

### Start / Resume the service (auto-scales 0 → 3 GPUs on demand)
```bash
# Default: min=0 (costs $0 when idle), max=3 parallel GPUs
uv run python -m scripts.manage_runpod start

# Optional: limit max parallel GPUs to 1 to stay strictly single-worker
uv run python -m scripts.manage_runpod start --max-workers 1
```
> **Note:** Because `workersMin=0` and `idleTimeout=10s`, you normally do **not** need to stop the endpoint between videos — RunPod automatically scales GPUs down to `0` ten seconds after your job finishes.

### Pause / Hard-Stop GPU billing (blocks all jobs, 0 GPUs allowed)
```bash
uv run python -m scripts.manage_runpod stop
```
Sets `workersMin=0, workersMax=0` and purges any queued jobs. GPU spend is guaranteed **\$0**. Your 60 GB Network Volume (with the 37.3 GB weights) remains preserved so restarting takes 1 second.

### Full Teardown (hit literal \$0.00/month when done with the project)
```bash
# Delete endpoint only (keeps the 60GB volume for instant redeploy later):
uv run python -m scripts.manage_runpod teardown

# Delete BOTH endpoint AND the 60GB Network Volume ($0.00/month total):
uv run python -m scripts.manage_runpod teardown --delete-volume
```

### Re-deploy anytime from scratch
```bash
uv run python -m scripts.deploy_runpod
uv run minimax-h3 warmup   # Needed only if you deleted the volume
```

---

## 3. Generating Videos via CLI

### A. Warmup / Verify Models & GPU
Run once after creating a fresh volume (downloads the 37.3 GB weights onto `/runpod-volume/models`) or anytime to verify GPU health:
```bash
uv run minimax-h3 warmup
```

### B. Generate a Short Multi-Reference Clip (5–15s, with Native Audio)
Pass your photo, outfit/dress photo, and background photo in order — they map automatically to **`<Picture 1>`**, **`<Picture 2>`**, **`<Picture 3>`** in your prompt:

```bash
uv run minimax-h3 clip \
  --name fashion_test \
  --ref-image ./me.jpg \
  --ref-image ./dress.jpg \
  --ref-image ./background.jpg \
  --duration 10 \
  --width 1152 \
  --height 640 \
  --steps 4 \
  --prompt "subject_definitions:
<Subject 1> is the person in <Picture 1>, wearing the exact dress from <Picture 2>, standing in the scene from <Picture 3>.

summary:
[reference generation] Two cinematic cuts of <Subject 1> wearing <Picture 2> inside <Picture 3>.

retention_analysis:
Preserve exact face and hair from <Picture 1>, exact dress fabric and color from <Picture 2>, and exact background lighting from <Picture 3>.

detailed_description:
[Shot 1: 0s-5s] Medium tracking shot. <Subject 1> walks smoothly forward and turns toward camera with a confident smile.
[Shot 2: 5s-10s] Crisp cut to close-up portrait as <Subject 1> speaks naturally to camera.

overall_soundscape: Natural ambient sound matching <Picture 3>, gentle footsteps, clear natural voice.

non_diegetic_music: Subtle warm modern piano underscore."
```
Output is saved to `outputs/single/01_clip.mp4` and `outputs/single/fashion_test_final.mp4` (plus `01_clip.last_frame.png` for chaining).

**Useful flags for `clip`:**
- `--ref-audio ./my_voice.wav` — attach optional voice/audio reference (`<Audio 1>`)
- `--first-frame ./start.png` — combine with `--ref-image` to run **`hybrid` mode** (anchors frame 0 **and** preserves all reference images!)
- `--ref-image-size max` — uses 2048px reference resolution for maximum face/dress detail (slower than default `match`)
- `--steps 6` or `--steps 8` — slightly sharper detail than 4-step default (still fast!)
- `--width 960 --height 544` — faster/cheaper draft renders (both must be multiples of 32; CLI auto-snaps)
- `--width 640 --height 1152` — vertical 9:16 Reels/Shorts/TikTok format

### C. Generate a Long 2–3 Minute Multi-Scene / Multi-Cut Video
1. Copy and edit [examples/multi_scene_storyboard.json](file:///Users/temurkhoshimov/myData/projects/runpod.io/examples/multi_scene_storyboard.json).
2. Add your shared `ref_images` once at the top (`<Picture 1>`, `<Picture 2>`, `<Picture 3>`).
3. Define as many 10–15s `scenes` as you want (e.g., 12–18 scenes = 2–3 minutes):
   - `"continue_from_previous": false` → runs in **parallel** across up to 3 RunPod GPUs (`ref2va` mode, hard cut to new angle/scene, zero identity drift).
   - `"continue_from_previous": true` → waits for previous scene, grabs its exact last frame, and runs **`hybrid` mode** (anchors previous last frame + keeps all original `ref_images`!).
4. Run the orchestrator:
```bash
uv run minimax-h3 storyboard examples/multi_scene_storyboard.json
```
Outputs:
- Per-scene clips: `outputs/multi_scene_demo/01_scene_01_establishing.mp4`, `02_...mp4`, etc.
- Final stitched MP4 with audio crossfade: `outputs/multi_scene_demo/multi_scene_demo_final.mp4`

### D. Re-Roll Only 1 Bad Scene (Without Re-Paying for the Whole Video)
If scene 2 looks slightly off, tweak its prompt/seed in the JSON and re-render **only** that scene (then automatically re-stitch):
```bash
uv run minimax-h3 storyboard examples/multi_scene_storyboard.json --only-scene scene_02_continuous_walk
```

---

## 4. Calling the Headless HTTP API Directly (from Any App / Backend)

Your RunPod Serverless endpoint URL is:
- **Submit job:** `POST https://api.runpod.ai/v2/iksf1j15rktqtl/run`
- **Poll status:** `GET https://api.runpod.ai/v2/iksf1j15rktqtl/status/{JOB_ID}`
- **Header:** `Authorization: Bearer <YOUR_RUNPOD_API_KEY>`

### Example `curl` (`ref2va` multi-reference with URLs)
```bash
curl -X POST "https://api.runpod.ai/v2/iksf1j15rktqtl/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "ref2va",
      "prompt": "subject_definitions:\n<Subject 1> is the person in <Picture 1> wearing <Picture 2> in <Picture 3>.\n\nsummary:\nTwo cinematic cuts of <Subject 1>.\n\ndetailed_description:\n[Shot 1: 0s-5s] Medium shot walking.\n[Shot 2: 5s-10s] Close-up smiling.\n\noverall_soundscape: Ambient breeze and footsteps.\nnon_diegetic_music: N/A",
      "ref_images": [
        {"url": "https://example.com/me.jpg"},
        {"url": "https://example.com/dress.jpg"},
        {"url": "https://example.com/bg.jpg"}
      ],
      "width": 1152,
      "height": 640,
      "duration_sec": 10.0,
      "steps": 4,
      "enable_sla": true,
      "return_last_frame": true
    }
  }'
```

### API Input Fields Summary
| Field | Values / Default | Purpose |
|---|---|---|
| `mode` | `"ref2va"`, `"hybrid"`, `"fl2va"`, `"warmup"` | Multi-ref, multi-ref + first/last frame, pure T2V/I2V, or health/sync |
| `prompt` | string | Use `<Picture 1>`, `<Picture 2>`, `<Audio 1>` + `[Shot 1]` / `[Shot 2]` cuts |
| `ref_images` | list of `{url}` or `{base64, filename}` (max 9) | Ordered reference images |
| `ref_audios` | list of `{url}` or `{base64, filename}` (max 3) | Optional voice/sound references |
| `first_frame` / `last_frame` | `{url}` or `{base64, filename}` | Optional keyframe anchors (`fl2va` or `hybrid`) |
| `width` / `height` | `1152` / `640` (multiples of 32) | 16:9 (`1152x640`, `960x544`, `1344x768`) or 9:16 (`640x1152`) |
| `duration_sec` | `1.0`–`16.0` (default `5.0`) | Auto-snapped to MiniMax-H3's `17k+5` frame grid at 24fps |
| `steps` | `4` (4–8 recommended for Turbo) | 4 is fastest; 6–8 adds extra crispness |
| `enable_sla` | `true` | Block-sparse attention (~2× faster) |
| `low_vram_chunk_ffn` | `false` | Set `true` if ever running on 16–24 GB GPUs |
| `upload_url` | optional presigned PUT URL | Upload MP4 directly to S3/R2 instead of returning base64 |
| `return_last_frame` | `true` | Returns `last_frame_base64` PNG for chaining |

### Optional S3 / Cloudflare R2 Storage
For clips over ~14 MB (or direct web delivery), add `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` to `.env` and re-run `uv run python -m scripts.deploy_runpod` — the worker automatically uploads MP4s to S3/R2 and returns `video.video_url`.

---

## 5. Prompting Tips for Best Face, Dress & Multi-Cut Accuracy

1. **Always use the MiniMax-H3 Reference Contract** (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) — it dramatically improves face & outfit consistency.
2. **Keep `<Picture N>` indices exact:** 1st image in `ref_images` = `<Picture 1>`, 2nd = `<Picture 2>`, 3rd = `<Picture 3>`.
3. **Multi-cut inside a single 10s clip:** Explicitly write timestamps and cut cues inside `detailed_description`:
   - `[Shot 1: 0s-4s] Wide establishing shot...`
   - `[Shot 2: 4s-7s] Hard cut to medium shot...`
   - `[Shot 3: 7s-10s] Close-up portrait...`
4. **Speed vs. quality:** Test prompts at `960x544`, `duration_sec=5`, `steps=4`, `ref_image_size="match"` first (cheapest & fastest), then render finals at `1152x640` or `1344x768` with `steps=6` (and `ref_image_size="max"` if needed).

---

## 6. Updating Worker Code Later

Any push to `main` touching `Dockerfile`, `docker/`, or `worker/` automatically triggers GitHub Actions to rebuild and push `ghcr.io/logos333/minimax-h3-worker:latest`. After the build finishes, restart workers by running:
```bash
uv run python -m scripts.deploy_runpod
```

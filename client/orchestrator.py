from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx

from client.interfaces import ServerlessRunner, VideoStitcher
from client.schemas import LocalOrRemoteAsset, SceneSpec, StoryboardSpec


def _encode_asset(asset: LocalOrRemoteAsset, base_dir: Path) -> dict[str, Any]:
    if asset.url:
        return {"url": asset.url}
    if asset.path:
        p = Path(asset.path)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Reference asset not found: {p}")
        raw = p.read_bytes()
        return {
            "base64": base64.b64encode(raw).decode("ascii"),
            "filename": p.name,
        }
    raise ValueError("LocalOrRemoteAsset must specify either 'path' or 'url'.")


async def _materialize_video(output: dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    video_meta = output.get("video") or {}
    if video_meta.get("video_base64"):
        raw = base64.b64decode(video_meta["video_base64"])
        await asyncio.to_thread(dest.write_bytes, raw)
        return dest
    if video_meta.get("video_url"):
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            resp = await client.get(video_meta["video_url"])
            resp.raise_for_status()
            await asyncio.to_thread(dest.write_bytes, resp.content)
        return dest
    raise RuntimeError(f"Worker response did not contain video_base64 or video_url: {output}")


class StoryboardOrchestrator:
    """Executes multi-scene storyboards: parallel for independent cuts, sequential hybrid for continuations."""

    def __init__(self, runner: ServerlessRunner, stitcher: VideoStitcher) -> None:
        self._runner = runner
        self._stitcher = stitcher

    async def warmup(self) -> dict[str, Any]:
        return await self._runner.run_job({"mode": "warmup"})

    async def run_storyboard(
        self,
        spec: StoryboardSpec,
        output_dir: Path,
        asset_base_dir: Path,
        only_scene: str | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        encoded_images = [_encode_asset(a, asset_base_dir) for a in spec.ref_images]
        encoded_audios = [_encode_asset(a, asset_base_dir) for a in spec.ref_audios]

        # Partition scenes into independent sequential chains (chains run in parallel)
        chains: list[list[tuple[int, SceneSpec]]] = []
        for idx, scene in enumerate(spec.scenes):
            if scene.continue_from_previous and chains:
                chains[-1].append((idx, scene))
            else:
                chains.append([(idx, scene)])

        clip_paths: dict[int, Path] = {}

        async def _run_chain(chain: list[tuple[int, SceneSpec]]) -> None:
            prev_last_frame_b64: str | None = None
            for idx, scene in chain:
                clip_path = output_dir / f"{idx + 1:02d}_{scene.id}.mp4"
                last_frame_file = output_dir / f"{idx + 1:02d}_{scene.id}.last_frame.png"

                if only_scene and scene.id != only_scene:
                    if clip_path.exists():
                        clip_paths[idx] = clip_path
                    if last_frame_file.exists():
                        prev_last_frame_b64 = base64.b64encode(last_frame_file.read_bytes()).decode("ascii")
                    continue

                scene_refs = (
                    [encoded_images[i] for i in scene.ref_image_indices]
                    if scene.ref_image_indices is not None
                    else encoded_images
                )

                mode = scene.mode
                first_frame_payload: dict[str, Any] | None = None
                if scene.continue_from_previous and prev_last_frame_b64:
                    first_frame_payload = {"base64": prev_last_frame_b64, "filename": "prev_last.png"}
                    if scene_refs:
                        mode = "hybrid"
                    else:
                        mode = "fl2va"
                elif scene.first_frame:
                    first_frame_payload = _encode_asset(scene.first_frame, asset_base_dir)

                last_frame_payload = (
                    _encode_asset(scene.last_frame, asset_base_dir) if scene.last_frame else None
                )

                payload: dict[str, Any] = {
                    "mode": mode,
                    "prompt": scene.prompt,
                    "ref_images": scene_refs if mode in ("ref2va", "hybrid") else [],
                    "ref_audios": encoded_audios if mode in ("ref2va", "hybrid") else [],
                    "ref_image_size": scene.ref_image_size,
                    "width": spec.width,
                    "height": spec.height,
                    "duration_sec": scene.duration_sec,
                    "fps": spec.fps,
                    "steps": scene.steps,
                    "enable_sla": spec.enable_sla,
                    "return_last_frame": True,
                }
                if first_frame_payload:
                    payload["first_frame"] = first_frame_payload
                if last_frame_payload:
                    payload["last_frame"] = last_frame_payload
                if scene.seed is not None:
                    payload["seed"] = scene.seed

                print(f"[orchestrator] Rendering scene {idx + 1}/{len(spec.scenes)} ({scene.id}, mode={mode})...", flush=True)
                result = await self._runner.run_job(payload)
                await _materialize_video(result, clip_path)
                clip_paths[idx] = clip_path

                prev_last_frame_b64 = result.get("last_frame_base64")
                if prev_last_frame_b64:
                    await asyncio.to_thread(
                        last_frame_file.write_bytes, base64.b64decode(prev_last_frame_b64)
                    )
                print(
                    f"[orchestrator] Finished {scene.id} -> {clip_path} ({result.get('elapsed_sec')}s)",
                    flush=True,
                )

        await asyncio.gather(*[_run_chain(c) for c in chains])

        ordered_clips = [clip_paths[i] for i in sorted(clip_paths.keys())]
        final_path = output_dir / f"{spec.name}_final.mp4"
        await self._stitcher.concat(
            clip_paths=ordered_clips,
            output_path=final_path,
            audio_crossfade_sec=spec.audio_crossfade_sec,
        )
        print(f"[orchestrator] Stitched final video -> {final_path}", flush=True)
        return final_path

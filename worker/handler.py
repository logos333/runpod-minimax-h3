from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import runpod

from worker.comfy_engine import ComfyUIVideoEngine
from worker.config import WorkerSettings, get_settings
from worker.interfaces import AssetResolver, ModelSynchronizer, OutputPublisher, VideoEngine
from worker.model_sync import HuggingFaceModelSynchronizer
from worker.schemas import GenerationMode, GenerationRequest, GenerationResponse
from worker.storage import (
    DefaultAssetResolver,
    DefaultOutputPublisher,
    extract_last_frame_base64,
)


class GenerationService:
    """Coordinates asset resolution, inference, output publishing, and temp cleanup."""

    def __init__(
        self,
        settings: WorkerSettings,
        engine: VideoEngine,
        resolver: AssetResolver,
        publisher: OutputPublisher,
        syncer: ModelSynchronizer,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._resolver = resolver
        self._publisher = publisher
        self._syncer = syncer

    async def execute(self, raw_input: dict[str, Any], job_id: str) -> GenerationResponse:
        t0 = time.monotonic()
        request = GenerationRequest.model_validate(raw_input)

        if request.mode == GenerationMode.WARMUP:
            models = await asyncio.to_thread(self._syncer.ensure_models)
            stats = await self._engine.health()
            return GenerationResponse(
                mode=GenerationMode.WARMUP,
                elapsed_sec=round(time.monotonic() - t0, 2),
                metadata={"models": models, "comfy_system": stats.get("system", {}), "devices": stats.get("devices", [])},
            )

        temp_files: list[Path] = []
        video_path: Path | None = None
        try:
            input_dir = self._settings.comfy_input_dir
            ref_image_names: list[str] = []
            for idx, asset in enumerate(request.ref_images):
                p = await self._resolver.resolve(asset, input_dir, f"{job_id}_img{idx}", ".png")
                temp_files.append(p)
                ref_image_names.append(p.name)

            ref_audio_names: list[str] = []
            for idx, asset in enumerate(request.ref_audios):
                p = await self._resolver.resolve(asset, input_dir, f"{job_id}_aud{idx}", ".wav")
                temp_files.append(p)
                ref_audio_names.append(p.name)

            first_frame_name: str | None = None
            if request.first_frame:
                p = await self._resolver.resolve(request.first_frame, input_dir, f"{job_id}_first", ".png")
                temp_files.append(p)
                first_frame_name = p.name

            last_frame_name: str | None = None
            if request.last_frame:
                p = await self._resolver.resolve(request.last_frame, input_dir, f"{job_id}_last", ".png")
                temp_files.append(p)
                last_frame_name = p.name

            video_path = await self._engine.generate(
                request=request,
                ref_image_names=ref_image_names,
                ref_audio_names=ref_audio_names,
                first_frame_name=first_frame_name,
                last_frame_name=last_frame_name,
                job_id=job_id,
            )

            published = await self._publisher.publish(video_path, upload_url=request.upload_url)
            last_frame_b64 = (
                await extract_last_frame_base64(video_path) if request.return_last_frame else None
            )

            return GenerationResponse(
                mode=request.mode,
                seed=request.seed,
                width=request.width,
                height=request.height,
                num_frames=request.aligned_frame_count(),
                fps=request.fps,
                elapsed_sec=round(time.monotonic() - t0, 2),
                video=published,
                last_frame_base64=last_frame_b64,
                metadata={
                    "steps": request.steps,
                    "enable_sla": request.enable_sla,
                    "ref_image_count": len(ref_image_names),
                    "ref_audio_count": len(ref_audio_names),
                },
            )
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)
            if video_path is not None:
                video_path.unlink(missing_ok=True)


def _build_service() -> GenerationService:
    settings = get_settings()
    return GenerationService(
        settings=settings,
        engine=ComfyUIVideoEngine(settings),
        resolver=DefaultAssetResolver(settings),
        publisher=DefaultOutputPublisher(settings),
        syncer=HuggingFaceModelSynchronizer(settings),
    )


_SERVICE = _build_service()


async def async_handler(event: dict[str, Any]) -> dict[str, Any]:
    job_id = str(event.get("id") or uuid.uuid4().hex[:12])
    raw_input = event.get("input") or {}
    response = await _SERVICE.execute(raw_input=raw_input, job_id=job_id)
    return response.model_dump(exclude_none=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": async_handler})

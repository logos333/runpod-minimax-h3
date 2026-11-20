from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from worker.config import WorkerSettings
from worker.interfaces import VideoEngine
from worker.schemas import GenerationRequest
from worker.workflow_builder import MiniMaxH3WorkflowBuilder


class ComfyUIVideoEngine(VideoEngine):
    """Async headless ComfyUI implementation of VideoEngine."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings
        self._builder = MiniMaxH3WorkflowBuilder(settings)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._settings.comfy_base_url}/system_stats")
            resp.raise_for_status()
            return resp.json()

    async def generate(
        self,
        request: GenerationRequest,
        ref_image_names: list[str],
        ref_audio_names: list[str],
        first_frame_name: str | None,
        last_frame_name: str | None,
        job_id: str,
    ) -> Path:
        client_id = uuid.uuid4().hex
        output_prefix = f"minimax_h3/{job_id}"
        graph, save_node_id = self._builder.build(
            request=request,
            ref_image_names=ref_image_names,
            ref_audio_names=ref_audio_names,
            first_frame_name=first_frame_name,
            last_frame_name=last_frame_name,
            output_prefix=output_prefix,
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            submit = await client.post(
                f"{self._settings.comfy_base_url}/prompt",
                json={"prompt": graph, "client_id": client_id},
            )
            if submit.status_code >= 400:
                raise RuntimeError(f"ComfyUI rejected prompt ({submit.status_code}): {submit.text}")
            prompt_id = submit.json()["prompt_id"]

            history = await self._wait_for_completion(client, prompt_id)

        return self._resolve_output_video(history, save_node_id)

    async def _wait_for_completion(
        self, client: httpx.AsyncClient, prompt_id: str
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._settings.comfy_timeout_sec
        poll_interval = 1.0

        while time.monotonic() < deadline:
            resp = await client.get(f"{self._settings.comfy_base_url}/history/{prompt_id}")
            resp.raise_for_status()
            payload = resp.json()
            if prompt_id in payload:
                entry = payload[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {messages}")
                if entry.get("outputs") or status.get("completed"):
                    return entry
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not finish within {self._settings.comfy_timeout_sec}s."
        )

    def _resolve_output_video(self, history: dict[str, Any], save_node_id: str) -> Path:
        outputs = history.get("outputs", {})
        node_out = outputs.get(save_node_id, {})

        # ComfyUI SaveVideo stores descriptors under 'images', 'videos', or 'gifs' depending on version
        for key in ("videos", "images", "gifs"):
            items = node_out.get(key) or []
            for item in items:
                filename = item.get("filename")
                if not filename:
                    continue
                subfolder = item.get("subfolder", "")
                candidate = self._settings.comfy_output_dir / subfolder / filename
                if candidate.exists():
                    return candidate

        raise FileNotFoundError(
            f"Could not locate rendered MP4 in ComfyUI outputs for node {save_node_id}: {node_out}"
        )

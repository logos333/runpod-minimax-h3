from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from client.config import ClientSettings
from client.interfaces import ServerlessRunner


class RunPodServerlessRunner(ServerlessRunner):
    """Async RunPod Serverless /run + /status client with bounded concurrency."""

    def __init__(self, settings: ClientSettings, endpoint_id: str | None = None) -> None:
        self._settings = settings
        self._api_key = settings.resolved_api_key()
        self._endpoint_id = endpoint_id or settings.runpod_endpoint_id
        if not self._endpoint_id:
            raise ValueError(
                "RUNPOD_ENDPOINT_ID is not set. Run scripts/deploy_runpod.py or set RUNPOD_ENDPOINT_ID in .env."
            )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def run_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = f"{self._settings.runpod_api_base.rstrip('/')}/{self._endpoint_id}"
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=60.0) as client:
                submit = await client.post(
                    f"{base}/run",
                    headers=self._headers,
                    json={"input": payload},
                )
                submit.raise_for_status()
                job_id = submit.json()["id"]
                print(f"[runpod] Submitted job {job_id}", flush=True)

                deadline = time.monotonic() + self._settings.job_timeout_sec
                while time.monotonic() < deadline:
                    resp = await client.get(f"{base}/status/{job_id}", headers=self._headers)
                    resp.raise_for_status()
                    data = resp.json()
                    status = data.get("status")

                    if status == "COMPLETED":
                        return data.get("output") or {}
                    if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                        raise RuntimeError(f"RunPod job {job_id} ended with {status}: {data}")

                    await asyncio.sleep(self._settings.poll_interval_sec)

                raise TimeoutError(
                    f"RunPod job {job_id} timed out after {self._settings.job_timeout_sec}s."
                )

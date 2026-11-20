from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ServerlessRunner(ABC):
    """Abstract client for invoking a remote serverless inference endpoint."""

    @abstractmethod
    async def run_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit an async job, poll until completion, and return the worker output payload."""


class VideoStitcher(ABC):
    """Abstract media stitcher combining per-scene MP4 clips into a final video."""

    @abstractmethod
    async def concat(
        self,
        clip_paths: list[Path],
        output_path: Path,
        audio_crossfade_sec: float = 0.12,
    ) -> Path:
        """Stitch ordered clips into output_path with clean A/V sync."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from worker.schemas import AssetInput, GenerationRequest, PublishedVideo


class ModelSynchronizer(ABC):
    """Ensures required model weights exist locally before inference."""

    @abstractmethod
    def ensure_models(self) -> dict[str, str]:
        """Download any missing weights and return a map of relative paths to resolved local paths."""


class AssetResolver(ABC):
    """Materializes remote URL / base64 inputs into local files readable by the engine."""

    @abstractmethod
    async def resolve(
        self,
        asset: AssetInput,
        target_dir: Path,
        default_prefix: str,
        default_ext: str,
    ) -> Path:
        """Save asset into target_dir and return the local Path."""


class OutputPublisher(ABC):
    """Publishes generated MP4 video bytes to S3/R2, presigned URL, or inline base64."""

    @abstractmethod
    async def publish(self, video_path: Path, upload_url: str | None = None) -> PublishedVideo:
        """Publish video file and return delivery descriptor."""


class VideoEngine(ABC):
    """Abstract video generation engine (decouples handler from ComfyUI/Diffusers/SGLang)."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return backend health/GPU diagnostics."""

    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest,
        ref_image_names: list[str],
        ref_audio_names: list[str],
        first_frame_name: str | None,
        last_frame_name: str | None,
        job_id: str,
    ) -> Path:
        """Execute generation and return path to the muxed audio+video MP4."""

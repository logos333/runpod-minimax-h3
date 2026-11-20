from __future__ import annotations

import random
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GenerationMode(StrEnum):
    """Supported execution modes on the stateless MiniMax-H3 worker."""

    WARMUP = "warmup"
    REF2VA = "ref2va"  # Multi-reference images/audio -> video + synced audio
    HYBRID = "hybrid"  # Multi-reference images + first/last-frame anchor via MiniMaxH3AddGuide
    FL2VA = "fl2va"    # Pure T2VA or First/Last-Frame-to-Video + synced audio


class AssetInput(BaseModel):
    """An input image or audio asset provided via URL or base64 payload."""

    url: str | None = Field(default=None, description="HTTP(S) URL to download asset from.")
    base64: str | None = Field(
        default=None, description="Raw base64 string or data URI (data:image/png;base64,...)."
    )
    filename: str | None = Field(
        default=None, description="Optional filename hint with extension (e.g. 'face.png')."
    )

    @model_validator(mode="after")
    def validate_source(self) -> AssetInput:
        if not self.url and not self.base64:
            raise ValueError("AssetInput requires either 'url' or 'base64'.")
        return self


class GenerationRequest(BaseModel):
    """Validated input schema for a single RunPod Serverless job."""

    mode: GenerationMode = Field(default=GenerationMode.REF2VA)
    prompt: str = Field(default="", description="Prompt (supports <Picture 1>, <Audio 1>, CUT 1/CUT 2, etc.).")

    # References for ref2va / hybrid (order maps directly to <Picture 1>, <Picture 2>, ...)
    ref_images: list[AssetInput] = Field(default_factory=list, max_length=9)
    ref_audios: list[AssetInput] = Field(default_factory=list, max_length=3)
    ref_image_size: Literal["match", "max"] = Field(
        default="match",
        description="'match' is fast (matches output area); 'max' uses 2048px short edge for max identity fidelity.",
    )

    # Keyframes & timeline soundtrack anchor
    first_frame: AssetInput | None = Field(default=None)
    last_frame: AssetInput | None = Field(default=None)
    guide_audio: AssetInput | None = Field(
        default=None,
        description="Timeline-locked soundtrack anchored via MiniMaxH3AddGuide for exact 1:1 lip-sync.",
    )
    guide_audio_frame_idx: int = Field(default=0, ge=-9999, le=9999)

    # Canvas & timing (MiniMax-H3 uses 32px multiples and 17k+5 frame grid @ 24fps)
    width: int = Field(default=1152, ge=256, le=1920)
    height: int = Field(default=640, ge=256, le=1920)
    duration_sec: float = Field(default=5.0, ge=1.0, le=16.0)
    fps: int = Field(default=24, ge=12, le=30)

    # Sampling overrides
    steps: int = Field(default=4, ge=1, le=50)
    seed: int = Field(default_factory=lambda: random.randint(1, 2**50))
    shift_video: float = Field(default=12.0, gt=0.0, le=100.0)
    shift_audio: float = Field(default=3.0, gt=0.0, le=100.0)
    enable_sla: bool = Field(default=True, description="Enable H3SLAAttention block-sparse speedup.")
    sla_sparsity: float = Field(default=0.9, ge=0.0, le=0.98)
    low_vram_chunk_ffn: bool = Field(
        default=False, description="Enable KJNodes MiniMaxChunkFeedForward for 16-24GB GPUs."
    )

    # Output delivery options
    upload_url: str | None = Field(
        default=None, description="Optional presigned HTTP PUT URL where the worker uploads the MP4."
    )
    return_last_frame: bool = Field(
        default=True,
        description="Also extract and return the last video frame as PNG base64 for chaining.",
    )

    @field_validator("width", "height")
    @classmethod
    def snap_to_multiple_of_32(cls, value: int) -> int:
        return max(256, round(value / 32) * 32)

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> GenerationRequest:
        if self.mode != GenerationMode.WARMUP and not self.prompt.strip():
            raise ValueError("'prompt' is required for generation modes.")
        if self.mode == GenerationMode.REF2VA and not (self.ref_images or self.ref_audios or self.guide_audio):
            raise ValueError(
                "mode='ref2va' requires at least one entry in 'ref_images', 'ref_audios', or 'guide_audio'."
            )
        if self.mode == GenerationMode.HYBRID and not (self.first_frame or self.last_frame):
            raise ValueError("mode='hybrid' requires 'first_frame' or 'last_frame' in addition to references.")
        return self

    def aligned_frame_count(self) -> int:
        """Align requested duration to MiniMax-H3's required 17k + 5 frame grid."""
        raw = max(5, round(self.duration_sec * self.fps))
        return raw + (5 - (raw % 17)) % 17


class PublishedVideo(BaseModel):
    """Published video artifact metadata."""

    delivery: Literal["presigned_url", "s3", "base64"]
    video_url: str | None = None
    video_base64: str | None = None
    size_bytes: int
    content_type: str = "video/mp4"


class GenerationResponse(BaseModel):
    """Structured output returned by the RunPod Serverless handler."""

    status: Literal["ok"] = "ok"
    mode: GenerationMode
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    elapsed_sec: float
    video: PublishedVideo | None = None
    last_frame_base64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

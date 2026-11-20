from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LocalOrRemoteAsset(BaseModel):
    """Reference image or audio supplied as local file path or HTTP(S) URL."""

    path: str | None = None
    url: str | None = None
    label: str | None = Field(
        default=None, description="Human label (e.g. 'Subject 1: user portrait', 'Dress reference')."
    )


class SceneSpec(BaseModel):
    """Single scene/shot in a multi-scene storyboard (typically 5-15s, may contain internal cuts)."""

    id: str = Field(..., description="Short scene slug (e.g. 'scene_01').")
    mode: Literal["ref2va", "hybrid", "fl2va"] = Field(default="ref2va")
    prompt: str = Field(..., description="MiniMax-H3 prompt referencing <Picture 1>, <Picture 2>, etc.")
    duration_sec: float = Field(default=10.0, ge=1.0, le=16.0)
    trim_frames: int | None = Field(
        default=None,
        ge=1,
        description="Exact integer frame count to keep at stitch time (eliminates 17k+5 grid drift).",
    )
    trim_duration_sec: float | None = Field(
        default=None,
        gt=0.0,
        le=16.0,
        description="Optional exact duration in seconds to trim this clip to during stitching.",
    )
    continue_from_previous: bool = Field(
        default=False,
        description="If true, waits for previous scene and passes its last frame as first_frame (hybrid mode).",
    )
    ref_image_indices: list[int] | None = Field(
        default=None,
        description="Optional subset of indices (0-based) into storyboard.ref_images; None uses all.",
    )
    ref_audio_indices: list[int] | None = Field(
        default=None,
        description="Optional subset of indices (0-based) into storyboard.ref_audios; None uses all.",
    )
    first_frame: LocalOrRemoteAsset | None = None
    last_frame: LocalOrRemoteAsset | None = None
    seed: int | None = None
    steps: int = Field(default=8, ge=1, le=50)
    ref_image_size: Literal["match", "max"] = Field(default="max")
    shift_video: float | None = Field(default=None, gt=0.0, le=100.0)
    shift_audio: float | None = Field(default=None, gt=0.0, le=100.0)
    enable_sla: bool | None = None
    sla_sparsity: float | None = Field(default=None, ge=0.0, le=0.98)
    lock_audio_timeline: bool | None = Field(
        default=None,
        description="Override StoryboardSpec.lock_audio_timeline for this scene.",
    )


class StoryboardSpec(BaseModel):
    """Multi-scene video project specification (supports short clips up to 2-3+ min videos)."""

    name: str = Field(default="project")
    width: int = Field(default=1152, ge=256, le=1920)
    height: int = Field(default=640, ge=256, le=1920)
    fps: int = Field(default=24, ge=12, le=30)
    enable_sla: bool = Field(default=False)
    sla_sparsity: float = Field(default=0.9, ge=0.0, le=0.98)
    shift_video: float = Field(default=8.0, gt=0.0, le=100.0)
    shift_audio: float = Field(default=3.0, gt=0.0, le=100.0)
    lock_audio_timeline: bool = Field(
        default=True,
        description="Anchor scene audio at frame_idx=0 via MiniMaxH3AddGuide for exact 1:1 lip-sync.",
    )
    audio_crossfade_sec: float = Field(default=0.0, ge=0.0, le=1.0)

    # Shared reference pack (order -> <Picture 1>, <Picture 2>, <Picture 3>, ...)
    ref_images: list[LocalOrRemoteAsset] = Field(default_factory=list)
    ref_audios: list[LocalOrRemoteAsset] = Field(default_factory=list)
    scenes: list[SceneSpec] = Field(..., min_length=1)


def build_minimax_reference_prompt(
    subject_definitions: str,
    summary: str,
    detailed_description: str,
    overall_soundscape: str,
    retention_analysis: str | None = None,
    non_diegetic_music: str = "N/A",
) -> str:
    """Format a prompt using the verified MiniMax-H3 Ref2VA contract for highest fidelity."""
    retention = retention_analysis or "All subjects, outfits, and backgrounds stay consistent with their reference pictures."
    return (
        f"subject_definitions:\n{subject_definitions.strip()}\n\n"
        f"summary:\n{summary.strip()}\n\n"
        f"retention_analysis:\n{retention.strip()}\n\n"
        f"detailed_description:\n{detailed_description.strip()}\n\n"
        f"overall_soundscape: {overall_soundscape.strip()}\n\n"
        f"non_diegetic_music: {non_diegetic_music.strip()}"
    )

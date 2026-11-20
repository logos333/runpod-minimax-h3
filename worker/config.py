from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ModelArtifactSpec:
    """Declarative specification for a single Hugging Face weight file."""

    repo_id: str
    remote_path: str
    local_relpath: str
    approx_gb: float


class WorkerSettings(BaseSettings):
    """Centralized configuration for the MiniMax-H3 Serverless worker."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Core directories
    models_root: Path = Field(default=Path("/runpod-volume/models"), alias="MODELS_ROOT")
    comfy_dir: Path = Field(default=Path("/app/ComfyUI"), alias="COMFY_DIR")
    comfy_host: str = Field(default="127.0.0.1", alias="COMFY_HOST")
    comfy_port: int = Field(default=8188, alias="COMFY_PORT")

    # Hugging Face & model filenames (MATLOWAI fused 4-step INT8 ConvRot + Comfy-Org VAEs/Text Encoder)
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    unet_repo: str = Field(
        default="MATLOWAI/minimax-h3-fused-turbo-int8-convrot", alias="MINIMAX_UNET_REPO"
    )
    unet_remote_path: str = Field(
        default="diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
        alias="MINIMAX_UNET_REMOTE_PATH",
    )
    unet_filename: str = Field(
        default="minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
        alias="MINIMAX_UNET_FILENAME",
    )

    comfy_assets_repo: str = Field(default="Comfy-Org/MiniMax-H3", alias="MINIMAX_ASSETS_REPO")
    text_encoder_filename: str = Field(
        default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", alias="MINIMAX_CLIP_FILENAME"
    )
    video_vae_filename: str = Field(
        default="minimax_h3_video_vae_int8_convrot.safetensors", alias="MINIMAX_VIDEO_VAE_FILENAME"
    )
    audio_vae_filename: str = Field(
        default="minimax_h3_audio_vae_fp32.safetensors", alias="MINIMAX_AUDIO_VAE_FILENAME"
    )

    # Default MiniMax-H3 sampling parameters (verified from MATLOWAI 4-step SLA recipes)
    default_steps: int = Field(default=4, alias="DEFAULT_STEPS")
    default_sampler: str = Field(default="res_multistep", alias="DEFAULT_SAMPLER")
    default_scheduler: str = Field(default="simple", alias="DEFAULT_SCHEDULER")
    default_shift_video: float = Field(default=12.0, alias="DEFAULT_SHIFT_VIDEO")
    default_shift_audio: float = Field(default=3.0, alias="DEFAULT_SHIFT_AUDIO")
    default_fps: int = Field(default=24, alias="DEFAULT_FPS")
    default_sla_sparsity: float = Field(default=0.9, alias="DEFAULT_SLA_SPARSITY")
    default_sla_block_size: str = Field(default="64", alias="DEFAULT_SLA_BLOCK_SIZE")
    default_sla_min_seq_len: int = Field(default=8192, alias="DEFAULT_SLA_MIN_SEQ_LEN")
    default_low_vram_chunk_ffn: bool = Field(default=False, alias="DEFAULT_LOW_VRAM_CHUNK_FFN")

    # Execution & transfer timeouts
    comfy_timeout_sec: float = Field(default=540.0, alias="COMFY_TIMEOUT_SEC")
    download_timeout_sec: float = Field(default=60.0, alias="DOWNLOAD_TIMEOUT_SEC")
    max_inline_base64_mb: float = Field(default=14.0, alias="MAX_INLINE_BASE64_MB")

    # Optional S3 / Cloudflare R2 output upload
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")
    s3_access_key_id: str | None = Field(default=None, alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = Field(default=None, alias="S3_SECRET_ACCESS_KEY")
    s3_public_base_url: str | None = Field(default=None, alias="S3_PUBLIC_BASE_URL")
    s3_region: str = Field(default="auto", alias="S3_REGION")

    @property
    def comfy_base_url(self) -> str:
        return f"http://{self.comfy_host}:{self.comfy_port}"

    @property
    def comfy_input_dir(self) -> Path:
        return self.comfy_dir / "input"

    @property
    def comfy_output_dir(self) -> Path:
        return self.comfy_dir / "output"

    def required_model_specs(self) -> list[ModelArtifactSpec]:
        return [
            ModelArtifactSpec(
                repo_id=self.unet_repo,
                remote_path=self.unet_remote_path,
                local_relpath=f"diffusion_models/{self.unet_filename}",
                approx_gb=19.54,
            ),
            ModelArtifactSpec(
                repo_id=self.comfy_assets_repo,
                remote_path=f"text_encoders/{self.text_encoder_filename}",
                local_relpath=f"text_encoders/{self.text_encoder_filename}",
                approx_gb=14.61,
            ),
            ModelArtifactSpec(
                repo_id=self.comfy_assets_repo,
                remote_path=f"vae/{self.video_vae_filename}",
                local_relpath=f"vae/{self.video_vae_filename}",
                approx_gb=2.62,
            ),
            ModelArtifactSpec(
                repo_id=self.comfy_assets_repo,
                remote_path=f"vae/{self.audio_vae_filename}",
                local_relpath=f"vae/{self.audio_vae_filename}",
                approx_gb=0.56,
            ),
        ]


_settings: WorkerSettings | None = None


def get_settings() -> WorkerSettings:
    global _settings
    if _settings is None:
        _settings = WorkerSettings()
    return _settings

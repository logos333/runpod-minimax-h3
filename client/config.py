from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_runpod_toml_key() -> str | None:
    cfg_path = Path.home() / ".runpod" / "config.toml"
    if not cfg_path.exists():
        return None
    try:
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        val = data.get("apikey")
        return str(val).strip() if val else None
    except Exception:
        return None


class ClientSettings(BaseSettings):
    """Centralized configuration for local CLI, orchestrator, and RunPod provisioning."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    runpod_api_key: str | None = Field(default=None, alias="RUNPOD_API_KEY")
    runpod_endpoint_id: str | None = Field(default=None, alias="RUNPOD_ENDPOINT_ID")
    runpod_api_base: str = Field(default="https://api.runpod.ai/v2", alias="RUNPOD_API_BASE")
    runpod_graphql_url: str = Field(
        default="https://api.runpod.io/graphql", alias="RUNPOD_GRAPHQL_URL"
    )

    worker_image: str = Field(
        default="ghcr.io/logos333/minimax-h3-worker:latest", alias="WORKER_IMAGE"
    )
    runpod_datacenter_id: str = Field(default="EU-RO-1", alias="RUNPOD_DATACENTER_ID")
    runpod_volume_name: str = Field(default="minimax-h3-models", alias="RUNPOD_VOLUME_NAME")
    runpod_volume_size_gb: int = Field(default=60, alias="RUNPOD_VOLUME_SIZE_GB")
    runpod_template_name: str = Field(
        default="minimax-h3-serverless-template", alias="RUNPOD_TEMPLATE_NAME"
    )
    runpod_endpoint_name: str = Field(
        default="minimax-h3-serverless", alias="RUNPOD_ENDPOINT_NAME"
    )
    runpod_gpu_ids: str = Field(
        default="ADA_32_PRO,AMPERE_80,BLACKWELL_96,ADA_48_PRO,ADA_24", alias="RUNPOD_GPU_IDS"
    )
    runpod_workers_min: int = Field(default=0, alias="RUNPOD_WORKERS_MIN")
    runpod_workers_max: int = Field(default=3, alias="RUNPOD_WORKERS_MAX")
    runpod_idle_timeout_sec: int = Field(default=10, alias="RUNPOD_IDLE_TIMEOUT_SEC")
    runpod_execution_timeout_ms: int = Field(
        default=600_000, alias="RUNPOD_EXECUTION_TIMEOUT_MS"
    )

    # Optional S3/R2 env pass-through into RunPod template
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")
    s3_access_key_id: str | None = Field(default=None, alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = Field(default=None, alias="S3_SECRET_ACCESS_KEY")
    s3_public_base_url: str | None = Field(default=None, alias="S3_PUBLIC_BASE_URL")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    poll_interval_sec: float = Field(default=3.0, alias="POLL_INTERVAL_SEC")
    job_timeout_sec: float = Field(default=900.0, alias="JOB_TIMEOUT_SEC")
    max_concurrency: int = Field(default=3, alias="MAX_CONCURRENCY")

    def resolved_api_key(self) -> str:
        key = self.runpod_api_key or _load_runpod_toml_key()
        if not key:
            raise ValueError(
                "RunPod API key not found. Set RUNPOD_API_KEY in .env or apikey in ~/.runpod/config.toml."
            )
        return key


def get_client_settings() -> ClientSettings:
    return ClientSettings()

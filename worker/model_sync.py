from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import hf_hub_download

from worker.config import ModelArtifactSpec, WorkerSettings, get_settings
from worker.interfaces import ModelSynchronizer


class HuggingFaceModelSynchronizer(ModelSynchronizer):
    """Downloads only the exact required MiniMax-H3 weight files onto the persistent volume."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings

    def ensure_models(self) -> dict[str, str]:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        models_root = self._settings.models_root
        models_root.mkdir(parents=True, exist_ok=True)

        lock_path = models_root / ".minimax_h3_sync.lock"
        resolved: dict[str, str] = {}

        with FileLock(str(lock_path), timeout=3600):
            for spec in self._settings.required_model_specs():
                target = self._ensure_single(spec, models_root)
                resolved[spec.local_relpath] = str(target)

        return resolved

    def _ensure_single(self, spec: ModelArtifactSpec, models_root: Path) -> Path:
        dest_path = models_root / spec.local_relpath
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Verify file exists and is non-trivial (>10MB)
        if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:
            size_gb = dest_path.stat().st_size / (1024**3)
            print(f"[model_sync] Cached: {spec.local_relpath} ({size_gb:.2f} GB)", flush=True)
            return dest_path

        print(
            f"[model_sync] Downloading {spec.repo_id}/{spec.remote_path} (~{spec.approx_gb:.2f} GB)...",
            flush=True,
        )
        t0 = time.monotonic()
        cache_dir = models_root / ".hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        downloaded = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=spec.remote_path,
                token=self._settings.hf_token,
                cache_dir=str(cache_dir),
            )
        )

        # Move real blob atomically (O(1) on same volume, zero double-disk peak)
        tmp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        if tmp_dest.exists():
            tmp_dest.unlink()
        real_blob = downloaded.resolve()
        shutil.move(str(real_blob), str(tmp_dest))
        os.replace(tmp_dest, dest_path)

        # Clean up dangling symlink in .hf_cache
        try:
            if downloaded.is_symlink() or downloaded.exists():
                downloaded.unlink()
        except OSError:
            pass

        elapsed = time.monotonic() - t0
        size_gb = dest_path.stat().st_size / (1024**3)
        print(
            f"[model_sync] Ready: {spec.local_relpath} ({size_gb:.2f} GB in {elapsed:.1f}s)",
            flush=True,
        )
        return dest_path


def main() -> None:
    settings = get_settings()
    syncer = HuggingFaceModelSynchronizer(settings)
    result = syncer.ensure_models()
    print(f"[model_sync] All {len(result)} model files verified.", flush=True)


if __name__ == "__main__":
    main()

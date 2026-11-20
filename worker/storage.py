from __future__ import annotations

import asyncio
import base64
import mimetypes
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import httpx

from worker.config import WorkerSettings
from worker.interfaces import AssetResolver, OutputPublisher
from worker.schemas import AssetInput, PublishedVideo


def _infer_extension(asset: AssetInput, content_type: str | None, default_ext: str) -> str:
    if asset.filename and "." in asset.filename:
        return "." + asset.filename.rsplit(".", 1)[-1].lower()
    if asset.url:
        path = urlparse(asset.url).path
        if "." in Path(path).name:
            return Path(path).suffix.lower()
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed
    return default_ext if default_ext.startswith(".") else f".{default_ext}"


class DefaultAssetResolver(AssetResolver):
    """Resolves HTTP(S) URLs and base64/data-URI payloads into local files."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._timeout = settings.download_timeout_sec

    async def resolve(
        self,
        asset: AssetInput,
        target_dir: Path,
        default_prefix: str,
        default_ext: str,
    ) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        content_type: str | None = None

        if asset.base64:
            raw_b64 = asset.base64.strip()
            if raw_b64.startswith("data:") and "," in raw_b64:
                header, raw_b64 = raw_b64.split(",", 1)
                if ";" in header:
                    content_type = header[5:].split(";", 1)[0]
            data = base64.b64decode(raw_b64)
        elif asset.url:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(asset.url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type")
                data = resp.content
        else:
            raise ValueError("AssetInput has neither url nor base64.")

        ext = _infer_extension(asset, content_type, default_ext)
        filename = f"{default_prefix}_{uuid.uuid4().hex[:10]}{ext}"
        dest = target_dir / filename
        await asyncio.to_thread(dest.write_bytes, data)
        return dest


class DefaultOutputPublisher(OutputPublisher):
    """Publishes MP4 via presigned PUT URL, S3/R2 bucket, or inline base64 fallback."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings

    async def publish(self, video_path: Path, upload_url: str | None = None) -> PublishedVideo:
        data = await asyncio.to_thread(video_path.read_bytes)
        size_bytes = len(data)

        if upload_url:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.put(
                    upload_url,
                    content=data,
                    headers={"Content-Type": "video/mp4"},
                )
                resp.raise_for_status()
            clean_url = upload_url.split("?", 1)[0]
            return PublishedVideo(
                delivery="presigned_url",
                video_url=clean_url,
                size_bytes=size_bytes,
            )

        if (
            self._settings.s3_bucket
            and self._settings.s3_access_key_id
            and self._settings.s3_secret_access_key
        ):
            url = await asyncio.to_thread(self._upload_s3, video_path)
            return PublishedVideo(delivery="s3", video_url=url, size_bytes=size_bytes)

        max_bytes = int(self._settings.max_inline_base64_mb * 1024 * 1024)
        if size_bytes > max_bytes:
            raise ValueError(
                f"Output video is {size_bytes / (1024**2):.2f} MB, exceeding inline base64 limit "
                f"({self._settings.max_inline_base64_mb:.1f} MB). Provide 'upload_url' or configure S3/R2."
            )

        encoded = base64.b64encode(data).decode("ascii")
        return PublishedVideo(
            delivery="base64",
            video_base64=encoded,
            size_bytes=size_bytes,
        )

    def _upload_s3(self, video_path: Path) -> str:
        s3 = boto3.client(
            "s3",
            endpoint_url=self._settings.s3_endpoint_url,
            aws_access_key_id=self._settings.s3_access_key_id,
            aws_secret_access_key=self._settings.s3_secret_access_key,
            region_name=self._settings.s3_region,
        )
        key = f"minimax-h3/{uuid.uuid4().hex}_{video_path.name}"
        s3.upload_file(
            str(video_path),
            self._settings.s3_bucket,
            key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        if self._settings.s3_public_base_url:
            return f"{self._settings.s3_public_base_url.rstrip('/')}/{key}"
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._settings.s3_bucket, "Key": key},
            ExpiresIn=86400,
        )


async def extract_last_frame_base64(video_path: Path) -> str | None:
    """Extract the final video frame as PNG base64 using FFmpeg for hybrid continuation."""
    out_png = video_path.with_suffix(".last_frame.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.2",
        "-i",
        str(video_path),
        "-update",
        "1",
        "-frames:v",
        "1",
        str(out_png),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    if proc.returncode != 0 or not out_png.exists():
        return None
    try:
        raw = await asyncio.to_thread(out_png.read_bytes)
        return base64.b64encode(raw).decode("ascii")
    finally:
        out_png.unlink(missing_ok=True)

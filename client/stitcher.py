from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from client.interfaces import VideoStitcher


class FFmpegVideoStitcher(VideoStitcher):
    """Concatenates scene clips into a unified MP4 with re-encoded H.264/AAC A/V sync."""

    async def concat(
        self,
        clip_paths: list[Path],
        output_path: Path,
        audio_crossfade_sec: float = 0.12,
    ) -> Path:
        if not clip_paths:
            raise ValueError("Cannot stitch an empty list of clips.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(clip_paths) == 1:
            shutil.copyfile(clip_paths[0], output_path)
            return output_path

        # Build filter_complex concat across all clips (uniform 24fps + 48kHz stereo AAC)
        cmd: list[str] = ["ffmpeg", "-y"]
        for p in clip_paths:
            cmd.extend(["-i", str(p)])

        filter_parts: list[str] = []
        concat_inputs: list[str] = []
        for idx in range(len(clip_paths)):
            filter_parts.append(f"[{idx}:v]fps=24,format=yuv420p,setsar=1[v{idx}]")
            if audio_crossfade_sec > 0:
                fade = f"afade=t=in:st=0:d={audio_crossfade_sec:.3f}"
                filter_parts.append(
                    f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,{fade}[a{idx}]"
                )
            else:
                filter_parts.append(
                    f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo[a{idx}]"
                )
            concat_inputs.append(f"[v{idx}][a{idx}]")

        filter_parts.append(
            f"{''.join(concat_inputs)}concat=n={len(clip_paths)}:v=1:a=1[outv][outa]"
        )

        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[outv]",
                "-map",
                "[outa]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg stitch failed: {stderr.decode('utf-8', errors='ignore')[-1200:]}")
        return output_path

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
        fps: int = 24,
        trim_frames: list[int | None] | None = None,
        audio_crossfade_sec: float = 0.0,
    ) -> Path:
        if not clip_paths:
            raise ValueError("Cannot stitch an empty list of clips.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        has_trims = bool(trim_frames and any(tf is not None for tf in trim_frames))
        if len(clip_paths) == 1 and not has_trims:
            shutil.copyfile(clip_paths[0], output_path)
            return output_path

        # Build filter_complex concat across all clips (uniform fps + exact frame trim + 48kHz stereo AAC)
        cmd: list[str] = ["ffmpeg", "-y"]
        for p in clip_paths:
            cmd.extend(["-i", str(p)])

        filter_parts: list[str] = []
        concat_inputs: list[str] = []
        for idx in range(len(clip_paths)):
            tf = trim_frames[idx] if trim_frames and idx < len(trim_frames) else None
            v_chain = f"fps={fps},format=yuv420p,setsar=1"
            a_chain = "aresample=48000,aformat=channel_layouts=stereo"
            if tf is not None:
                dur_exact = tf / float(fps)
                v_chain += f",trim=end_frame={tf},setpts=PTS-STARTPTS"
                a_chain += f",atrim=duration={dur_exact:.6f},asetpts=PTS-STARTPTS"
            if audio_crossfade_sec > 0:
                a_chain += f",afade=t=in:st=0:d={audio_crossfade_sec:.3f}"

            filter_parts.append(f"[{idx}:v]{v_chain}[v{idx}]")
            filter_parts.append(f"[{idx}:a]{a_chain}[a{idx}]")
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

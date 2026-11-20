from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from client.config import get_client_settings
from client.orchestrator import StoryboardOrchestrator
from client.runpod_client import RunPodServerlessRunner
from client.schemas import LocalOrRemoteAsset, SceneSpec, StoryboardSpec
from client.stitcher import FFmpegVideoStitcher


async def _cmd_warmup(args: argparse.Namespace) -> None:
    settings = get_client_settings()
    orch = StoryboardOrchestrator(
        runner=RunPodServerlessRunner(settings, endpoint_id=args.endpoint_id),
        stitcher=FFmpegVideoStitcher(),
    )
    res = await orch.warmup()
    print(json.dumps(res, indent=2))


async def _cmd_clip(args: argparse.Namespace) -> None:
    settings = get_client_settings()
    refs = [
        LocalOrRemoteAsset(url=r) if r.startswith(("http://", "https://")) else LocalOrRemoteAsset(path=r)
        for r in (args.ref_image or [])
    ]
    audios = [
        LocalOrRemoteAsset(url=a) if a.startswith(("http://", "https://")) else LocalOrRemoteAsset(path=a)
        for a in (args.ref_audio or [])
    ]
    first_frame = None
    if args.first_frame:
        first_frame = (
            LocalOrRemoteAsset(url=args.first_frame)
            if args.first_frame.startswith(("http://", "https://"))
            else LocalOrRemoteAsset(path=args.first_frame)
        )

    mode = args.mode
    if mode == "auto":
        if refs and first_frame:
            mode = "hybrid"
        elif refs or audios:
            mode = "ref2va"
        else:
            mode = "fl2va"

    spec = StoryboardSpec(
        name=args.name,
        width=args.width,
        height=args.height,
        ref_images=refs,
        ref_audios=audios,
        scenes=[
            SceneSpec(
                id="clip",
                mode=mode,
                prompt=args.prompt,
                duration_sec=args.duration,
                first_frame=first_frame,
                steps=args.steps,
                seed=args.seed,
                ref_image_size=args.ref_image_size,
            )
        ],
    )
    orch = StoryboardOrchestrator(
        runner=RunPodServerlessRunner(settings, endpoint_id=args.endpoint_id),
        stitcher=FFmpegVideoStitcher(),
    )
    await orch.run_storyboard(spec, output_dir=Path(args.output_dir), asset_base_dir=Path.cwd())


async def _cmd_storyboard(args: argparse.Namespace) -> None:
    settings = get_client_settings()
    sb_path = Path(args.file).resolve()
    spec = StoryboardSpec.model_validate(json.loads(sb_path.read_text(encoding="utf-8")))
    out_dir = Path(args.output_dir or f"outputs/{spec.name}")
    orch = StoryboardOrchestrator(
        runner=RunPodServerlessRunner(settings, endpoint_id=args.endpoint_id),
        stitcher=FFmpegVideoStitcher(),
    )
    await orch.run_storyboard(
        spec=spec,
        output_dir=out_dir,
        asset_base_dir=sb_path.parent,
        only_scene=args.only_scene,
        resume=args.resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax-H3 RunPod Serverless CLI")
    parser.add_argument("--endpoint-id", default=None, help="Override RUNPOD_ENDPOINT_ID")
    sub = parser.add_subparsers(dest="command", required=True)

    p_warm = sub.add_parser("warmup", help="Sync weights onto volume & verify GPU health")
    p_warm.set_defaults(func=_cmd_warmup)

    p_clip = sub.add_parser("clip", help="Generate a single 5-15s video+audio clip")
    p_clip.add_argument("--prompt", required=True)
    p_clip.add_argument("--ref-image", action="append", help="Local path or URL (repeatable: <Picture 1>, <Picture 2>...)")
    p_clip.add_argument("--ref-audio", action="append", help="Optional local audio path or URL (<Audio 1>...)")
    p_clip.add_argument("--first-frame", default=None, help="Optional first frame path or URL")
    p_clip.add_argument("--mode", choices=["auto", "ref2va", "hybrid", "fl2va"], default="auto")
    p_clip.add_argument("--duration", type=float, default=5.0)
    p_clip.add_argument("--width", type=int, default=1152)
    p_clip.add_argument("--height", type=int, default=640)
    p_clip.add_argument("--steps", type=int, default=4)
    p_clip.add_argument("--seed", type=int, default=None)
    p_clip.add_argument("--ref-image-size", choices=["match", "max"], default="match")
    p_clip.add_argument("--name", default="test_clip")
    p_clip.add_argument("--output-dir", default="outputs/single")
    p_clip.set_defaults(func=_cmd_clip)

    p_sb = sub.add_parser("storyboard", help="Generate a multi-scene (up to 2-3+ min) video from JSON")
    p_sb.add_argument("file", help="Path to storyboard JSON")
    p_sb.add_argument("--output-dir", default=None)
    p_sb.add_argument("--only-scene", default=None, help="Re-render only one scene ID and re-stitch")
    p_sb.add_argument("--resume", action="store_true", help="Skip already-rendered non-empty scene clips")
    p_sb.set_defaults(func=_cmd_storyboard)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()

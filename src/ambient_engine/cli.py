from __future__ import annotations

import argparse
import json
from pathlib import Path

from ambient_engine.app import AmbientEngine
from ambient_engine.profiles.loader import list_profiles


def main() -> None:
    parser = argparse.ArgumentParser(prog="ambient", description="Free/open ambient production engine")
    parser.add_argument("--project-root", default=".", help="Project root containing config/ and profiles/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render a full session")
    render_parser.add_argument("--profile", required=True)
    render_parser.add_argument("--target-length", default=None)
    render_parser.add_argument("--runtime", default="cpu-safe", choices=["cpu-safe", "gpu"])
    render_parser.add_argument("--seed", default=42, type=int)
    render_parser.add_argument("--dry-run", action="store_true")
    render_parser.add_argument("--background-image", default=None)
    render_parser.add_argument("--with-shorts", action="store_true")

    qc_parser = subparsers.add_parser("qc", help="Run QC on an existing session")
    qc_parser.add_argument("--session", required=True)

    package_parser = subparsers.add_parser("package", help="Package/report an existing session")
    package_parser.add_argument("--session", required=True)

    shorts_parser = subparsers.add_parser("shorts", help="Generate 3 vertical Shorts from a finished session")
    shorts_parser.add_argument("--session", required=True)

    publish_parser = subparsers.add_parser("publish", help="Publish or dry-run a publish")
    publish_parser.add_argument("target", choices=["youtube"])
    publish_parser.add_argument("--session", required=True)
    publish_parser.add_argument("--dry-run", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="Run the canonical demo profile")
    demo_parser.add_argument("--profile", default="afterblue_sleep")
    demo_parser.add_argument("--target-length", default="2h")
    demo_parser.add_argument("--runtime", default="cpu-safe", choices=["cpu-safe", "gpu"])
    demo_parser.add_argument("--seed", default=42, type=int)
    demo_parser.add_argument("--dry-run", action="store_true")
    demo_parser.add_argument("--background-image", default=None)
    demo_parser.add_argument("--with-shorts", action="store_true")

    profiles_parser = subparsers.add_parser("profiles", help="List installed profiles")

    args = parser.parse_args()
    engine = AmbientEngine(Path(args.project_root).resolve())

    if args.command == "render":
        result = engine.render(
            profile_id=args.profile,
            target_length=args.target_length,
            runtime_mode=args.runtime,
            seed=args.seed,
            dry_run=args.dry_run,
            background_image_path=Path(args.background_image).resolve() if args.background_image else None,
            with_shorts=args.with_shorts,
        )
    elif args.command == "qc":
        result = engine.run_qc(Path(args.session))
    elif args.command == "package":
        result = engine.package(Path(args.session))
    elif args.command == "shorts":
        result = engine.generate_shorts(Path(args.session))
    elif args.command == "publish":
        result = engine.publish(Path(args.session), dry_run=args.dry_run)
    elif args.command == "demo":
        result = engine.render(
            profile_id=args.profile,
            target_length=args.target_length,
            runtime_mode=args.runtime,
            seed=args.seed,
            dry_run=args.dry_run,
            background_image_path=Path(args.background_image).resolve() if args.background_image else None,
            with_shorts=args.with_shorts,
        )
    else:
        profiles = list_profiles(Path(args.project_root).resolve() / "profiles")
        result = {"profiles": profiles}

    print(json.dumps(result, indent=2, ensure_ascii=False))

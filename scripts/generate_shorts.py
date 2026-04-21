from __future__ import annotations

import argparse
import json
from pathlib import Path

from ambient_engine.app import AmbientEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3 portrait Shorts from an existing ambient session.")
    parser.add_argument("--project-root", default=".", help="Project root containing config/ and profiles/")
    parser.add_argument("--session", required=True, help="Absolute or relative path to the finished session directory.")
    args = parser.parse_args()

    engine = AmbientEngine(Path(args.project_root).resolve())
    result = engine.generate_shorts(Path(args.session))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

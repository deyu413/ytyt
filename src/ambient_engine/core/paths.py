from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    plans: Path
    sections: Path
    stems: Path
    exports: Path
    reports: Path
    temp: Path
    manifests: Path
    logs: Path

    def ensure(self) -> "SessionPaths":
        for path in (
            self.root,
            self.plans,
            self.sections,
            self.stems,
            self.exports,
            self.reports,
            self.temp,
            self.manifests,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


class ProjectPaths:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.config_dir = self.root / "config"
        self.profiles_dir = self.root / "profiles"
        self.docs_dir = self.root / "docs"
        self.sessions_dir = self.root / "sessions"
        self.tests_dir = self.root / "tests"

    def create_session(self, session_id: str) -> SessionPaths:
        base = self.sessions_dir / session_id
        return SessionPaths(
            root=base,
            plans=base / "plans",
            sections=base / "sections",
            stems=base / "stems",
            exports=base / "exports",
            reports=base / "reports",
            temp=base / "temp",
            manifests=base / "manifests",
            logs=base / "logs",
        ).ensure()

    @staticmethod
    def first_existing(paths: Iterable[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None


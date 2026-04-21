from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SessionManifest:
    session_id: str
    profile_id: str
    runtime_mode: str
    seed: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provider_capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_routing: list[dict[str, Any]] = field(default_factory=list)
    section_plan: list[dict[str, Any]] = field(default_factory=list)
    asset_lineage: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    variation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SessionManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


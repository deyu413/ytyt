from __future__ import annotations

import json
from pathlib import Path

from ambient_engine.core.durations import humanize_seconds
from ambient_engine.core.manifest import SessionManifest


def write_qc_report(metrics: dict[str, object], score_card: dict[str, object], gate: dict[str, object], output_path: Path) -> Path:
    payload = {
        "metrics": metrics,
        "score_card": score_card,
        "gate": gate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_session_report(
    manifest: SessionManifest,
    metadata: dict[str, object],
    outputs: dict[str, str],
    output_path: Path,
) -> Path:
    plan_lines = []
    for section in manifest.section_plan:
        plan_lines.append(
            f"- `{section['role']}` · {humanize_seconds(section['duration_seconds'])} · provider `{section['provider']}`"
        )
    qc = manifest.qc
    qc_lines = [
        f"- Accepted: `{qc.get('accepted', False)}`",
        f"- Global Score: `{qc.get('global_score', 'n/a')}`",
        f"- True Peak: `{qc.get('true_peak', 'n/a')}`",
        f"- Loudness: `{qc.get('integrated_loudness', 'n/a')}`",
        f"- Reference DNA Score: `{qc.get('reference_dna_score', 'n/a')}`",
        f"- Bass Anchor Score: `{qc.get('bass_anchor_score', 'n/a')}`",
        f"- Dynamic Breath Score: `{qc.get('dynamic_breath_score', 'n/a')}`",
        f"- Low-mid Boxiness Risk: `{qc.get('lowmid_boxiness_risk', 'n/a')}`",
    ]
    report = "\n".join(
        [
            f"# Session Report · {manifest.session_id}",
            "",
            "## Summary",
            f"- Profile: `{manifest.profile_id}`",
            f"- Runtime: `{manifest.runtime_mode}`",
            f"- Seed: `{manifest.seed}`",
            f"- Title: {metadata['title']}",
            "",
            "## Section Plan",
            *plan_lines,
            "",
            "## QC",
            *qc_lines,
            "",
            "## Outputs",
            *[f"- `{name}`: `{path}`" for name, path in outputs.items()],
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path

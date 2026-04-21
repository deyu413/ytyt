from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DurationSpec:
    raw: str
    seconds: int


def parse_duration(value: str | int | float) -> DurationSpec:
    if isinstance(value, (int, float)):
        return DurationSpec(raw=str(value), seconds=int(float(value)))
    text = str(value).strip().lower()
    if not text:
        raise ValueError("Duration cannot be empty.")
    if text.endswith("ms"):
        return DurationSpec(raw=text, seconds=max(1, int(float(text[:-2]) / 1000.0)))
    if text.endswith("s"):
        return DurationSpec(raw=text, seconds=int(float(text[:-1])))
    if text.endswith("m"):
        return DurationSpec(raw=text, seconds=int(float(text[:-1]) * 60))
    if text.endswith("h"):
        return DurationSpec(raw=text, seconds=int(float(text[:-1]) * 3600))
    if ":" in text:
        parts = [int(part) for part in text.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return DurationSpec(raw=text, seconds=minutes * 60 + seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return DurationSpec(raw=text, seconds=hours * 3600 + minutes * 60 + seconds)
        raise ValueError(f"Unsupported duration format: {value}")
    return DurationSpec(raw=text, seconds=int(float(text)))


def humanize_seconds(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


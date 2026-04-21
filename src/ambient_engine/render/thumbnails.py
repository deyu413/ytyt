from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ambient_engine.core.durations import humanize_seconds
from ambient_engine.profiles.schema import Profile


def render_thumbnail(
    profile: Profile,
    metadata: dict[str, object],
    target_seconds: int,
    static_frame_path: Path,
    output_path: Path,
    variation: dict[str, object] | None = None,
    size: tuple[int, int] = (1280, 720),
) -> Path:
    variation = variation or {}
    base = Image.open(static_frame_path).convert("RGBA").resize(size)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_small = _font(26)
    accent = profile.thumbnail_style.get("accent", "#8fd0ff")
    accent_rgb = _hex(accent)
    label = str(metadata["thumbnail_text"]).upper()
    duration = humanize_seconds(target_seconds).upper()

    subject_alignment = str(variation.get("subject_alignment", "center"))
    right_stack = subject_alignment == "left"
    panel = (676, 54, 1220, 666) if right_stack else (60, 54, 604, 666)
    title_x = panel[0] + 34
    title_y = panel[1] + 70
    panel_overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel_overlay)
    panel_draw.rounded_rectangle(panel, radius=28, fill=(4, 7, 15, 132), outline=accent_rgb + (168,), width=3)
    panel_draw.line((panel[0] + 28, panel[1] + 168, panel[2] - 28, panel[1] + 168), fill=accent_rgb + (110,), width=2)
    panel_draw.line((panel[0] + 28, panel[3] - 96, panel[2] - 28, panel[3] - 96), fill=(255, 255, 255, 48), width=1)

    title_lines = _wrap_label(label, max_chars=12 if right_stack else 11)
    font_large = _fit_font(title_lines[:3], max_width=panel[2] - panel[0] - 68, start_size=88)
    y = title_y
    for line in title_lines[:3]:
        panel_draw.text((title_x, y), line, font=font_large, fill=(255, 255, 255, 244))
        y += font_large.size + 10

    panel_draw.text(
        (title_x, panel[1] + 190 + min(2, len(title_lines) - 1) * 90),
        profile.branding.get("series_name", "Ambient Series").upper(),
        font=font_small,
        fill=accent_rgb + (220,),
    )
    panel_draw.text((title_x, panel[3] - 78), duration, font=font_small, fill=(255, 255, 255, 184))
    panel_draw.text(
        (panel[2] - 206, panel[3] - 78),
        str(metadata["hud_label"]).upper(),
        font=font_small,
        fill=(255, 255, 255, 154),
    )

    result = Image.alpha_composite(base, panel_overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def _wrap_label(label: str, max_chars: int) -> list[str]:
    words = label.split()
    if not words:
        return [label]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _fit_font(lines: list[str], max_width: int, start_size: int) -> ImageFont.ImageFont:
    for size in range(start_size, 51, -4):
        font = _font(size)
        if all(font.getbbox(line)[2] <= max_width for line in lines if line):
            return font
    return _font(52)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex(value: str) -> tuple[int, int, int]:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))

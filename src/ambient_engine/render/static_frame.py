from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ambient_engine.profiles.schema import Profile


def render_static_frame(
    profile: Profile,
    variation: dict[str, object],
    output_path: Path,
    size: tuple[int, int] = (1920, 1080),
    background_image_path: Path | None = None,
) -> Path:
    width, height = size
    style = profile.thumbnail_style
    top = _hex(style.get("gradient_top", "#10263c"))
    bottom = _hex(style.get("gradient_bottom", "#04070c"))
    accent = _hex(style.get("accent", "#8fd0ff"))

    if background_image_path is not None and background_image_path.exists():
        image = _render_from_background(background_image_path, size=size, top=top, bottom=bottom, accent=accent)
    else:
        image = _render_generated_background(size=size, variation=variation, top=top, bottom=bottom, accent=accent)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _render_from_background(
    background_image_path: Path,
    size: tuple[int, int],
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> Image.Image:
    width, height = size
    source = Image.open(background_image_path).convert("RGBA")
    cover = ImageOps.fit(source, size, method=Image.Resampling.LANCZOS)
    source_ratio = source.width / max(1, source.height)
    target_ratio = width / max(1, height)
    portrait_style = source_ratio < target_ratio * 0.82

    background = cover.filter(ImageFilter.GaussianBlur(radius=26)) if portrait_style else cover.copy()

    tint = Image.new("RGBA", size, (0, 0, 0, 0))
    tint_draw = ImageDraw.Draw(tint)
    for y in range(height):
        ratio = y / max(1, height - 1)
        mix = tuple(int(top[i] * (1.0 - ratio) + bottom[i] * ratio) for i in range(3))
        alpha = int((84 + 40 * ratio) if portrait_style else (58 + 34 * ratio))
        tint_draw.line([(0, y), (width, y)], fill=mix + (alpha,))
    image = Image.alpha_composite(background, tint)

    if portrait_style:
        foreground = source.copy()
        foreground.thumbnail((int(width * 0.58), int(height * 0.97)), Image.Resampling.LANCZOS)
        shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        shadow_x = int(width * 0.07)
        shadow_y = height - foreground.height
        shadow_box = (shadow_x + 16, shadow_y + 18, shadow_x + foreground.width + 16, shadow_y + foreground.height + 18)
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle(shadow_box, radius=36, fill=(0, 0, 0, 135))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=28))
        image = Image.alpha_composite(image, shadow)

        panel = Image.new("RGBA", size, (0, 0, 0, 0))
        panel_draw = ImageDraw.Draw(panel)
        foreground_box = (shadow_x - 22, shadow_y - 26, shadow_x + foreground.width + 24, shadow_y + foreground.height + 18)
        panel_draw.rounded_rectangle(foreground_box, radius=42, fill=(4, 7, 14, 88), outline=accent + (78,), width=2)
        image = Image.alpha_composite(image, panel)
        image.alpha_composite(foreground, (shadow_x, shadow_y))
    else:
        subject_glow = Image.new("RGBA", size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(subject_glow)
        glow_draw.ellipse(
            (int(width * 0.06), int(height * 0.07), int(width * 0.67), int(height * 0.98)),
            fill=accent + (18,),
            outline=accent + (0,),
        )
        subject_glow = subject_glow.filter(ImageFilter.GaussianBlur(radius=72))
        image = Image.alpha_composite(image, subject_glow)

    image = Image.alpha_composite(image, _vignette(size=size, accent=accent))
    image = Image.alpha_composite(image, _hud_support_overlay(size=size, accent=accent))
    return image


def _render_generated_background(
    size: tuple[int, int],
    variation: dict[str, object],
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = tuple(int(top[i] * (1.0 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color + (255,))

    alignment = variation.get("subject_alignment", "center")
    orb_x = {"left": int(width * 0.22), "center": width // 2, "right": int(width * 0.78)}.get(alignment, width // 2)
    orb_y = int(height * 0.42)
    orb = Image.new("RGBA", size, (0, 0, 0, 0))
    orb_draw = ImageDraw.Draw(orb)
    orb_draw.ellipse(
        (orb_x - 180, orb_y - 180, orb_x + 180, orb_y + 180),
        fill=accent + (48,),
        outline=accent + (140,),
        width=3,
    )
    orb = orb.filter(ImageFilter.GaussianBlur(radius=48))
    image = Image.alpha_composite(image, orb)

    grid_color = accent + (26,)
    for x in range(80, width, 120):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(60, height, 120):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    image = Image.alpha_composite(image, _hud_support_overlay(size=size, accent=accent))
    return image


def _vignette(size: tuple[int, int], accent: tuple[int, int, int]) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(
        (int(width * 0.11), int(height * 0.03), int(width * 0.89), int(height * 1.08)),
        outline=accent + (28,),
        width=2,
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=120))
    dark = Image.new("RGBA", size, (0, 0, 0, 0))
    dark_draw = ImageDraw.Draw(dark)
    dark_draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 32))
    return Image.alpha_composite(dark, overlay)


def _hud_support_overlay(size: tuple[int, int], accent: tuple[int, int, int]) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    panel_box = (88, height - 244, width - 88, height - 58)
    draw.rounded_rectangle(panel_box, radius=30, fill=(2, 5, 11, 94), outline=accent + (44,), width=2)
    draw.line((132, height - 228, width - 132, height - 228), fill=accent + (42,), width=1)
    bottom_fade = Image.new("RGBA", size, (0, 0, 0, 0))
    fade_draw = ImageDraw.Draw(bottom_fade)
    for y in range(height - 280, height):
        ratio = (y - (height - 280)) / 280.0
        fade_draw.line([(0, y), (width, y)], fill=(0, 0, 0, int(18 + 74 * ratio)))
    return Image.alpha_composite(bottom_fade, overlay)


def _hex(value: str) -> tuple[int, int, int]:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))

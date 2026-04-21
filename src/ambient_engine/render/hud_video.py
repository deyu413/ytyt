from __future__ import annotations

import subprocess
from pathlib import Path


def render_hud_video(
    ffmpeg_executable: str | None,
    static_frame_path: Path,
    audio_path: Path,
    output_path: Path,
    hud_label: str,
) -> Path:
    del hud_label
    if not ffmpeg_executable:
        raise RuntimeError("FFmpeg executable not available. Install imageio-ffmpeg or ffmpeg.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_attempts = [
        (
            "[0:v]format=rgba,drawbox=x=110:y=836:w=1700:h=188:color=0x02050A@0.24:t=fill,"
            "drawbox=x=110:y=836:w=1700:h=188:color=0x9FD9FF@0.12:t=2[bg];"
            "[1:a]asplit=2[a1][a2];"
            "[a1]showfreqs=s=1620x118:mode=bar:fscale=log:ascale=sqrt:colors=0x9FD9FF,"
            "format=rgba,colorchannelmixer=aa=0.74[freq];"
            "[a2]showwaves=s=1620x44:mode=line:colors=0xFFFFFF,"
            "format=rgba,colorchannelmixer=aa=0.84[waves];"
            "[bg][freq]overlay=150:874[tmp];"
            "[tmp][waves]overlay=150:986[v]"
        ),
        (
            "[1:a]showwaves=s=1920x180:mode=cline:colors=0x9FD9FF@0.85,format=rgba[w];"
            "[0:v][w]overlay=0:900[v]"
        ),
    ]

    for filter_graph in filter_attempts:
        cmd = [
            ffmpeg_executable,
            "-y",
            "-loop",
            "1",
            "-i",
            str(static_frame_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path

    raise RuntimeError(f"HUD video render failed: {result.stderr[-400:]}")

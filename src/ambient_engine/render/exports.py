from __future__ import annotations

import subprocess
from pathlib import Path

import soundfile as sf


def export_preview(master_path: Path, output_path: Path, seconds: int, sample_rate: int) -> Path:
    frames = seconds * sample_rate
    with sf.SoundFile(master_path, mode="r") as reader:
        excerpt = reader.read(frames, dtype="float32", always_2d=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, excerpt, sample_rate, subtype="PCM_24")
    return output_path


def export_mp3(ffmpeg_executable: str | None, input_path: Path, output_path: Path, bitrate: str = "320k") -> Path:
    if not ffmpeg_executable:
        raise RuntimeError("FFmpeg executable not available. Install imageio-ffmpeg or ffmpeg.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(input_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MP3 export failed: {result.stderr[-400:]}")
    return output_path


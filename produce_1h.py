"""
MusicaYT — Script de Producción: 1 Hora de audio + video
=========================================================
Usa los segmentos RAW existentes, los extiende a 1h,
aplica postprocesado (LUFS, 432Hz, limiter) y ensambla
con el video loop.
"""
import sys
import time
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import yaml
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Cargar config
with open('config/settings.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
with open('config/prompts/trading_ambient.yaml', 'r', encoding='utf-8') as f:
    prompts = yaml.safe_load(f)

SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TARGET_HOURS = 1.0
VIDEO_LOOP = Path(r"c:\Users\nerod\OneDrive\Desktop\allprojects\musciayt\assets\video_loops\0419(1).mp4")

print("=" * 60)
print(f"🚀 PRODUCCIÓN: {TARGET_HOURS}h de Trading Focus Audio + Video")
print(f"   Session: {SESSION_ID}")
print(f"   Video:   {VIDEO_LOOP.name} (1080p 30fps, listo)")
print("=" * 60)

total_start = time.time()

# ═══════════════════════════════════
# FASE 1: Extender audio a 1h
# ═══════════════════════════════════
print(f"\n{'═'*50}")
print("🎵 FASE 1: Extender audio a 1 hora")
print(f"{'═'*50}")

from src.audio.extender import AudioExtender
extender = AudioExtender(config)

# Usar todos los segmentos RAW disponibles
segments = sorted(Path('assets/audio_raw').glob('session_*_0*.wav'))
print(f"  Segmentos disponibles: {len(segments)}")

t0 = time.time()
extended_path = extender.extend_to_duration(
    segments, target_hours=TARGET_HOURS,
    output_filename=f"{SESSION_ID}_extended"
)
t1 = time.time()

data, sr = sf.read(str(extended_path))
print(f"  ✅ Extensión completada en {t1-t0:.1f}s")
print(f"     Duración: {len(data)/sr/60:.1f} min | SR: {sr}Hz | Peak: {np.max(np.abs(data)):.3f}")

# ═══════════════════════════════════
# FASE 2: Post-procesamiento
# ═══════════════════════════════════
print(f"\n{'═'*50}")
print("🔧 FASE 2: Post-procesamiento (LUFS + 432Hz + Limiter)")
print(f"{'═'*50}")

from src.audio.postprocess import AudioPostProcessor
pp = AudioPostProcessor(config)

t0 = time.time()
master_path = pp.process_full_pipeline(
    extended_path, texture_paths=None,
    output_filename=f"{SESSION_ID}_master"
)
t1 = time.time()

data_m, sr_m = sf.read(str(master_path))
print(f"  ✅ Masterización completada en {t1-t0:.1f}s")
print(f"     Duración: {len(data_m)/sr_m/60:.1f} min | Peak: {np.max(np.abs(data_m)):.3f}")

# ═══════════════════════════════════
# FASE 3: Metadata + Thumbnail
# ═══════════════════════════════════
print(f"\n{'═'*50}")
print("📝 FASE 3: Metadata + Thumbnail")
print(f"{'═'*50}")

from src.assembly.metadata import MetadataGenerator
from src.assembly.thumbnail import ThumbnailGenerator

meta_gen = MetadataGenerator(config, prompts)
prompt_info = {'category': 'stoic_night', 'mood': 'calm_focused', 'visual_style': 'night_cityscape'}

metadata = meta_gen.generate_metadata(
    TARGET_HOURS, prompt_info,
    title_family='aesthetic',
    description_mode='long'
)

print(f"  Título: {metadata['title']}")
print(f"  Familia: {metadata['title_family']}")
print(f"  Label: {metadata['thumbnail_label']}")

# Extraer frame del video para thumbnail
import subprocess
frame_path = Path('output') / f"{SESSION_ID}_frame.png"
subprocess.run([
    'ffmpeg', '-y', '-ss', '5', '-i', str(VIDEO_LOOP),
    '-vframes', '1', '-vf', 'scale=1280:720',
    str(frame_path)
], capture_output=True)

thumb_gen = ThumbnailGenerator(config)
style = meta_gen._mood_to_style('calm_focused')
thumb_path = thumb_gen.generate(
    TARGET_HOURS, 'calm_focused', style,
    video_frame_path=str(frame_path) if frame_path.exists() else None,
    output_filename=f"{SESSION_ID}_thumb",
    label=metadata['thumbnail_label']
)
print(f"  ✅ Thumbnail: {thumb_path}")

# Guardar metadata JSON
import json
meta_json = Path('output') / f"{SESSION_ID}_metadata.json"
with open(meta_json, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════
# FASE 4: Ensamblaje Video
# ═══════════════════════════════════
print(f"\n{'═'*50}")
print("📦 FASE 4: Ensamblaje de Video (loop + audio)")
print(f"{'═'*50}")

output_video = Path('output') / f"{SESSION_ID}_FINAL.mp4"
audio_duration = len(data_m) / sr_m

print(f"  Audio: {audio_duration/60:.1f} min")
print(f"  Video loop: {VIDEO_LOOP.name} (21.7s, 1080p 30fps)")
print(f"  Output: {output_video}")
print(f"  ⏳ Esto tardará unos minutos...")

t0 = time.time()

# FFmpeg: loop video 1080p y mezclar con audio
cmd = [
    'ffmpeg', '-y',
    '-stream_loop', '-1',           # Repetir video infinitamente
    '-i', str(VIDEO_LOOP),          # Video input (loop)
    '-i', str(master_path),         # Audio input
    '-t', str(audio_duration),      # Duración = audio
    '-map', '0:v:0',               # Fuerza el video del primer input
    '-map', '1:a:0',               # Fuerza el audio del segundo input (master WAV)
    '-c:v', 'h264_nvenc',           # GPU Encode (NVENC)
    '-preset', 'slow',              # High quality NVENC preset
    '-cq', '20',                    # Constante quality for NVENC (equivalent to CRF)
    '-c:a', 'aac',                 # Audio AAC
    '-b:a', '192k',                # Bitrate audio
    '-pix_fmt', 'yuv420p',         # Compatibilidad YouTube
    '-movflags', '+faststart',     # Fast start para streaming
    '-shortest',                    # Terminar con el stream más corto
    str(output_video)
]

process = subprocess.run(cmd, capture_output=True, text=True)
t1 = time.time()

if process.returncode == 0:
    size_mb = output_video.stat().st_size / (1024 * 1024)
    print(f"  ✅ Video ensamblado en {t1-t0:.0f}s ({(t1-t0)/60:.1f} min)")
    print(f"     Tamaño: {size_mb:.0f} MB")
else:
    print(f"  ❌ Error FFmpeg:")
    print(process.stderr[-500:])

# ═══════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════
total_time = time.time() - total_start
print(f"\n{'='*60}")
print(f"📊 PRODUCCIÓN COMPLETADA en {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'='*60}")
print(f"  🎵 Audio master:  {master_path}")
print(f"  📹 Video final:   {output_video}")
print(f"  🖼️  Thumbnail:     {thumb_path}")
print(f"  📝 Metadata:      {meta_json}")
print(f"  📋 Título:        {metadata['title']}")
print(f"\n  📌 SIGUIENTE PASO:")
print(f"     Subir a YouTube:")
print(f"     python src/orchestrator.py --assemble-only --audio {master_path} --dry-run")
print(f"{'='*60}")

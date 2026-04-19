"""
MusicaYT — Script de Audio para SageMaker Studio Lab
=====================================================
Script optimizado para la GPU T4 (16GB VRAM) de AWS SageMaker Studio Lab.

USO:
  1. Abrir SageMaker Studio Lab (https://studiolab.sagemaker.aws/)
  2. Iniciar sesión GPU (4h máximo)
  3. Subir este script
  4. Ejecutar: python notebook_audio.py
  5. Descargar los archivos generados (o sincronizar con Google Drive)

LIMITACIONES:
  - 4h por sesión GPU, 8h/día
  - 15GB almacenamiento persistente
  - T4 GPU (16GB VRAM) — YuEGP Profile 1
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# ══════════════════════════════════════════
# CONFIGURACIÓN — EDITAR ANTES DE USAR
# ══════════════════════════════════════════

YUEGP_REPO = "https://github.com/YuEGP/YuEGP.git"
YUEGP_DIR = Path.home() / "YuEGP"  # Se instala en home para persistencia
OUTPUT_DIR = Path.home() / "audio_output"
PROFILE = 1  # Profile 1 = 16GB VRAM (T4)

# Prompts para generación (copia de trading_ambient.yaml)
PROMPTS = [
    {
        "name": "stoic_night_01",
        "text": "Deep focus ambient electronic, dark atmospheric synth pads, "
                "60 BPM, key of A minor, minimal percussion, slow evolving textures, "
                "cinematic drone, no vocals, meditation undertone, professional studio quality"
    },
    {
        "name": "war_room_01",
        "text": "Lo-fi electronic ambient, subtle data processing sounds, "
                "70 BPM, atmospheric pads, gentle glitch textures, "
                "the sound of a high-tech trading floor at night, "
                "monitors humming, keyboard clicks in the distance, "
                "professional and sleek, no distracting melodies"
    },
    {
        "name": "alpha_01",
        "text": "432Hz binaural beat ambient foundation, continuous drone tone, "
                "subtle harmonic overtones, no rhythm, no melody, "
                "pure sonic texture for deep concentration, laboratory quality, "
                "theta to alpha wave transition design"
    },
    {
        "name": "cyberpunk_01",
        "text": "Cyberpunk ambient, neon-lit atmosphere, deep bass undertone, "
                "65 BPM, F minor, synthetic textures resembling digital data streams, "
                "subtle rhythmic pulse like a heartbeat monitor, "
                "futuristic trading terminal aesthetic"
    },
]


def setup_environment():
    """Instala YuEGP si no está disponible."""
    print("🔧 Verificando entorno...")
    
    if not YUEGP_DIR.exists():
        print("📦 Instalando YuEGP...")
        subprocess.run(
            ["git", "clone", YUEGP_REPO, str(YUEGP_DIR)],
            check=True
        )
    
    # Instalar dependencias
    req_file = YUEGP_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True
        )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Entorno listo")


def check_gpu():
    """Verifica disponibilidad de GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        print(f"🖥️  GPU: {result.stdout.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ No se detectó GPU. ¿Iniciaste una sesión GPU?")
        return False


def generate_segment(prompt_info, segment_duration=30):
    """Genera un segmento de audio con YuEGP."""
    name = prompt_info["name"]
    text = prompt_info["text"]
    output_path = OUTPUT_DIR / f"{name}_{datetime.now().strftime('%H%M%S')}.wav"

    print(f"\n🎵 Generando: {name}")
    print(f"   Prompt: {text[:80]}...")

    cmd = [
        sys.executable, "-m", "yuegp.generate",
        "--prompt", text,
        "--duration", str(segment_duration),
        "--output", str(output_path),
        "--profile", str(PROFILE),
        "--sample_rate", "44100",
    ]

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd=str(YUEGP_DIR)
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"   ✅ Generado en {elapsed:.0f}s: {output_path.name}")
            return output_path
        else:
            print(f"   ❌ Error: {result.stderr[:200]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout tras 30 min")
        return None


def estimate_session_capacity():
    """Estima cuántos segmentos se pueden generar en una sesión de 4h."""
    # T4 con Profile 1: ~5-8 min por 30s de audio
    est_time_per_seg = 8 * 60  # 8 minutos en segundos (conservador)
    session_time = 4 * 60 * 60  # 4 horas en segundos
    safety_margin = 0.85  # 85% del tiempo (margen para setup y download)
    
    capacity = int((session_time * safety_margin) / est_time_per_seg)
    print(f"\n📊 Capacidad estimada de la sesión:")
    print(f"   ~{capacity} segmentos de 30s ({capacity * 30 / 60:.0f} min de audio)")
    print(f"   Tiempo por segmento: ~{est_time_per_seg/60:.0f} min")
    return capacity


def main():
    """Script principal para SageMaker Studio Lab."""
    print("=" * 60)
    print("🚀 MusicaYT — Generación de Audio en SageMaker")
    print("=" * 60)
    
    if not check_gpu():
        print("\nInicia una sesión GPU en SageMaker Studio Lab e intenta de nuevo.")
        sys.exit(1)
    
    setup_environment()
    capacity = estimate_session_capacity()
    
    session_start = datetime.now()
    session_end = session_start + timedelta(hours=3, minutes=45)  # 15 min margen
    
    print(f"\n⏰ Sesión: {session_start.strftime('%H:%M')} → {session_end.strftime('%H:%M')}")
    print(f"   (3h45m de generación, 15 min de margen)")
    
    generated = []
    prompt_idx = 0
    
    while datetime.now() < session_end:
        prompt = PROMPTS[prompt_idx % len(PROMPTS)]
        
        remaining = (session_end - datetime.now()).total_seconds() / 60
        print(f"\n⏳ Tiempo restante: {remaining:.0f} min")
        
        if remaining < 10:
            print("⚠️  Menos de 10 min restantes, deteniendo generación")
            break
        
        result = generate_segment(prompt, segment_duration=30)
        if result:
            generated.append(str(result))
        
        prompt_idx += 1
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"📊 SESIÓN COMPLETADA")
    print(f"   Segmentos generados: {len(generated)}")
    print(f"   Audio total: {len(generated) * 30 / 60:.0f} min")
    print(f"   Archivos en: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Listar archivos para descarga
    if generated:
        print("\n📁 Archivos para descargar:")
        for f in generated:
            size = Path(f).stat().st_size / (1024 * 1024)
            print(f"   • {Path(f).name} ({size:.1f} MB)")
        
        # Guardar log
        log = {
            "session": session_start.isoformat(),
            "segments": len(generated),
            "files": generated,
            "total_minutes": len(generated) * 30 / 60
        }
        log_path = OUTPUT_DIR / f"session_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"\n📝 Log guardado: {log_path}")


if __name__ == "__main__":
    main()

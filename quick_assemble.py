"""
Quick Assembly — Usa segmentos existentes para crear pista de 2h.
NO usa GPU — solo CPU (numpy/soundfile).
"""
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    # Cargar config
    with open("config/settings.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open("config/prompts/trading_ambient.yaml", 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"

    # Crear directorios
    for d in ['assets/audio_final', 'output', 'output/thumbnails', 'logs']:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1. Encontrar segmentos existentes
    raw_dir = Path("assets/audio_raw")
    segments = sorted(raw_dir.glob("session_*.wav"))
    # Incluir también el test
    test_seg = raw_dir / "test_primer_audio.wav"
    if test_seg.exists() and test_seg not in segments:
        segments.append(test_seg)

    logger.info(f"Encontrados {len(segments)} segmentos de audio:")
    for s in segments:
        logger.info(f"  • {s.name}")

    if len(segments) < 2:
        logger.error("Necesitas al menos 2 segmentos. Genera más primero.")
        return

    # 2. Extender a 2 horas
    logger.info("\n" + "=" * 50)
    logger.info("🔁 EXTENSIÓN A 2 HORAS")
    logger.info("=" * 50)

    from src.audio.extender import AudioExtender
    extender = AudioExtender(config)
    extended_path = extender.extend_to_duration(
        segments, 
        target_hours=2.0, 
        output_filename=f"{session_id}_extended"
    )
    logger.info(f"✅ Extendido: {extended_path}")

    # 3. Post-procesamiento (LUFS, 432Hz, limiter) — SIN texturas (no hay GPU)
    logger.info("\n" + "=" * 50)
    logger.info("🎚️ POST-PROCESAMIENTO")
    logger.info("=" * 50)

    from src.audio.postprocess import AudioPostProcessor
    pp = AudioPostProcessor(config)
    master_path = pp.process_full_pipeline(
        extended_path,
        texture_paths=None,  # Sin texturas por ahora
        output_filename=f"{session_id}_master"
    )
    logger.info(f"✅ Master: {master_path}")

    # 4. Metadata SEO
    logger.info("\n" + "=" * 50)
    logger.info("📝 METADATA + THUMBNAIL")
    logger.info("=" * 50)

    prompt_info = {
        'text': 'stoic night ambient',
        'mood': 'calm_focused',
        'energy': 'low',
        'category': 'stoic_night',
        'visual_style': 'night_cityscape'
    }

    from src.assembly.metadata import MetadataGenerator
    meta_gen = MetadataGenerator(config, prompts_data)
    metadata = meta_gen.generate_metadata(2.0, prompt_info)
    logger.info(f"  Título: {metadata['title']}")

    # Guardar metadata
    meta_json = Path("output") / f"{session_id}_metadata.json"
    with open(meta_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 5. Thumbnail
    from src.assembly.thumbnail import ThumbnailGenerator
    thumb_gen = ThumbnailGenerator(config)
    style = meta_gen._mood_to_style(prompt_info['mood'])
    thumb = thumb_gen.generate(2.0, prompt_info['mood'], style,
                                output_filename=f"{session_id}_thumb")
    logger.info(f"  Thumbnail: {thumb}")

    # 6. Video prompts
    logger.info("\n" + "=" * 50)
    logger.info("🎬 PROMPTS DE VIDEO (PARA VEO3)")
    logger.info("=" * 50)

    from src.assembly.video_prompts import VideoPromptGenerator
    vpg = VideoPromptGenerator(config)
    pair = vpg.generate_prompt_pair(prompt_info['mood'])
    formatted = vpg.format_for_user(pair, prompt_info['mood'], prompt_info['category'])

    prompts_file = Path("output") / f"{session_id}_video_prompts.txt"
    with open(prompts_file, 'w', encoding='utf-8') as f:
        f.write(formatted)

    # Resumen
    print("\n" + "=" * 60)
    print("📊 ¡SESIÓN COMPLETADA!")
    print("=" * 60)
    print(f"  🎵 Audio master: {master_path}")
    print(f"  🖼️  Thumbnail:    {thumb}")
    print(f"  📝 Metadata:     {meta_json}")
    print(f"  🎬 Prompts Veo3: {prompts_file}")
    print(f"\n  📌 SIGUIENTE PASO:")
    print(f"     1. Copia los prompts de {prompts_file}")
    print(f"     2. Genera 2 clips en Veo3")
    print(f"     3. Coloca en assets/video_loops/")
    print(f"     4. Ejecuta: python src/orchestrator.py --assemble-only --audio {master_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

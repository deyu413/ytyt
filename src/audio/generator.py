"""
MusicaYT — Generador de Audio (ACE-Step)
==========================================
Usa la API Python directa de ACE-Step para generar segmentos
de audio ambient de alta calidad.

ACE-Step: ~3-4GB VRAM, genera hasta 4 min por segmento.
RTX 3050 8GB: OK con cpu_offload activado.

El modelo se descarga automáticamente la primera vez (~3GB).
"""

import os
import sys
import random
import logging
import time
from pathlib import Path
from datetime import datetime

import yaml
import torch

logger = logging.getLogger(__name__)


def load_config():
    """Carga la configuración global."""
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_prompts():
    """Carga la biblioteca de prompts del nicho actual."""
    config = load_config()
    niche = config['project']['niche']
    prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts" / f"{niche}_ambient.yaml"
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class AudioGenerator:
    """Generador de audio usando ACE-Step directamente."""

    def __init__(self, engine=None):
        self.config = load_config()
        self.prompts_data = load_prompts()
        self.engine = engine or self.config['audio']['primary_engine']
        self.audio_config = self.config['audio']
        self.output_dir = Path(self.config['paths']['audio_raw'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline se inicializa lazy (primera llamada a generate)
        self._pipeline = None

        logger.info(f"AudioGenerator inicializado — motor: {self.engine}")

    def _init_pipeline(self):
        """Inicializa el pipeline ACE-Step (lazy loading)."""
        if self._pipeline is not None:
            return

        logger.info("Cargando modelo ACE-Step (primera vez descarga ~3GB)...")

        # Asegurar que ACE-Step está en el path
        ace_path = self.audio_config.get('ace_step', {}).get('model_path', '')
        if ace_path and ace_path not in sys.path:
            sys.path.insert(0, ace_path)

        from acestep.pipeline_ace_step import ACEStepPipeline

        low_vram = self.audio_config.get('ace_step', {}).get('low_vram_mode', True)

        self._pipeline = ACEStepPipeline(
            checkpoint_dir=ace_path if ace_path else None,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
            torch_compile=False,
            cpu_offload=False,          # False = GPU directa (8GB es suficiente)
            overlapped_decode=True,     # Reduce VRAM en decodificación
        )

        logger.info("ACE-Step cargado y listo")

    def select_prompt(self, category=None, mood=None):
        """
        Selecciona un prompt de la biblioteca.

        Returns:
            dict con text, mood, energy, category, visual_style
        """
        categories = self.prompts_data['categories']

        if category and category in categories:
            cat_data = categories[category]
        else:
            cat_name = random.choice(list(categories.keys()))
            cat_data = categories[cat_name]
            category = cat_name

        prompts = cat_data['prompts']
        if mood:
            filtered = [p for p in prompts if p.get('mood') == mood]
            if filtered:
                prompts = filtered

        selected = random.choice(prompts)
        logger.info(f"Prompt — Categoría: {category}, Mood: {selected.get('mood', 'N/A')}")

        return {
            'text': selected['text'].strip(),
            'mood': selected.get('mood', 'unknown'),
            'energy': selected.get('energy', 'low'),
            'category': category,
            'visual_style': cat_data.get('visual_style', 'night_cityscape')
        }

    def generate_segment(self, prompt_text, output_filename=None, duration_sec=None):
        """
        Genera un segmento de audio.

        Args:
            prompt_text: Prompt descriptivo para la generación
            output_filename: Nombre del archivo (sin extensión)
            duration_sec: Duración en segundos (max 240)

        Returns:
            Path al archivo .wav generado
        """
        # Inicializar pipeline si no lo está
        self._init_pipeline()

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"segment_{timestamp}"

        duration = duration_sec or self.audio_config['segment_duration_sec']
        # ACE-Step soporta hasta 240s (4 min)
        duration = min(duration, 240)

        output_path = self.output_dir / f"{output_filename}.wav"

        logger.info(f"Generando — Duración: {duration}s")
        logger.info(f"  Prompt: {prompt_text[:80]}...")

        start_time = time.time()

        try:
            # Llamar a ACE-Step directamente
            result = self._pipeline(
                audio_duration=float(duration),
                prompt=prompt_text,
                lyrics="",                     # Sin letra (instrumental)
                infer_step=25,                 # 25 pasos = buena calidad para ambient (más rápido)
                guidance_scale=15.0,           # CFG scale
                scheduler_type="euler",
                cfg_type="apg",                # APG = mejor calidad
                omega_scale=10.0,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=False,           # No hay lyrics
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=str(output_path),
                batch_size=1,
            )

            elapsed = time.time() - start_time
            logger.info(f"  ✅ Generado en {elapsed:.1f}s → {output_path.name}")

            # Limpiar VRAM
            self._pipeline.cleanup_memory()

            if output_path.exists():
                return output_path
            else:
                # ACE-Step puede guardar con sufijo diferente — buscar
                parent = output_path.parent
                possible = list(parent.glob(f"{output_filename}*"))
                if possible:
                    actual = possible[0]
                    logger.info(f"  Archivo encontrado como: {actual.name}")
                    return actual
                raise FileNotFoundError(f"Audio no generado: {output_path}")

        except Exception as e:
            logger.error(f"Error generando audio: {e}")
            raise

    def generate_batch(self, num_segments, category=None, prefix="batch"):
        """
        Genera un lote de segmentos de audio.

        Returns:
            Lista de dicts con {path, prompt, index}
        """
        generated = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Batch: {num_segments} segmentos...")

        for i in range(num_segments):
            prompt_info = self.select_prompt(category=category)
            filename = f"{prefix}_{timestamp}_{i:03d}"

            try:
                path = self.generate_segment(
                    prompt_text=prompt_info['text'],
                    output_filename=filename
                )
                generated.append({
                    'path': path,
                    'prompt': prompt_info,
                    'index': i
                })
                logger.info(f"Segmento {i+1}/{num_segments} completado")

            except Exception as e:
                logger.error(f"Error en segmento {i+1}: {e}")
                continue

        logger.info(f"Batch: {len(generated)}/{num_segments} segmentos generados")
        return generated


# --- CLI ---
if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--category', default=None, help='Categoría de prompt')
    @click.option('--duration', default=None, type=int, help='Duración en segundos (max 240)')
    @click.option('--batch', default=1, type=int, help='Número de segmentos')
    @click.option('--output', default=None, help='Nombre del archivo')
    def main(category, duration, batch, output):
        """Genera audio para MusicaYT con ACE-Step."""
        # Configurar logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/audio_generator.log', encoding='utf-8')
            ]
        )

        gen = AudioGenerator()

        if batch > 1:
            results = gen.generate_batch(batch, category=category)
            print(f"\n✓ {len(results)} segmentos generados:")
            for r in results:
                print(f"  • {r['path']}")
        else:
            prompt_info = gen.select_prompt(category=category)
            path = gen.generate_segment(
                prompt_text=prompt_info['text'],
                output_filename=output,
                duration_sec=duration
            )
            print(f"\n✓ Audio: {path}")

    main()

"""
MusicaYT — Orchestrator Maestro v2
====================================
Pipeline centrado en MÚSICA. El video es manual (Veo3).

FLUJO:
1. Genera audio (ACE-Step local o YuEGP SageMaker)
2. Extiende audio a duración objetivo (2-10h)
3. Genera/reutiliza texturas ambientales
4. Post-procesa (LUFS, 432Hz, crossfade, limiter, mezcla texturas)
5. Genera prompts de video para Veo3
6. Genera metadata SEO + thumbnail
7. [Cuando hay clips de Veo3] Ensambla video final
8. Sube a YouTube (o dry-run)

MODOS:
  python src/orchestrator.py --music-only            # Solo genera audio + metadata
  python src/orchestrator.py --full                   # Pipeline completo (requiere clips de video)
  python src/orchestrator.py --full --dry-run         # Completo sin subir a YouTube
  python src/orchestrator.py --assemble-only          # Solo ensamblar (audio ya existe)
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def setup_logging(config):
    """Logging centralizado."""
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding='utf-8')
        ]
    )
    return log_file


def load_configs():
    """Carga configuraciones."""
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    niche = config['project']['niche']
    prompts_path = base_dir / "config" / "prompts" / f"{niche}_ambient.yaml"
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f)

    return config, prompts_data


class Pipeline:
    """Pipeline de producción centrado en música."""

    def __init__(self, config, prompts_data):
        self.config = config
        self.prompts_data = prompts_data
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{self.timestamp}"
        self.results = {}

        # Crear directorios necesarios
        for key in ['audio_raw', 'audio_final', 'video_loops', 'output', 'logs']:
            Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════
    # MODO 1: SOLO MÚSICA
    # Genera audio + metadata + prompts de video
    # ═══════════════════════════════════════
    def run_music_only(self, duration_hours=None, category=None, num_segments=None):
        """
        Genera solo la música y los assets asociados.
        NO ensambla video ni sube a YouTube.
        
        Ideal para:
        - Primer uso: generar audio mientras preparas los clips de Veo3
        - Batch: generar muchas pistas para publicar luego
        """
        self._print_banner("MUSIC-ONLY MODE")
        
        target_hours = duration_hours or self.config['audio']['default_duration_hours']
        prompt_info = self._select_prompt(category)

        # Fase Audio
        audio_path = self._phase_audio(prompt_info, num_segments, target_hours)

        # Fase Metadata
        metadata, thumb_path = self._phase_metadata(
            prompt_info, target_hours,
            title_family=getattr(self, '_title_family', None),
            description_mode=getattr(self, '_desc_mode', 'long')
        )

        # Fase Prompts de Video
        video_prompts = self._phase_video_prompts(prompt_info)

        self.results.update({
            'mode': 'music_only',
            'audio': str(audio_path),
            'thumbnail': str(thumb_path),
            'metadata': metadata,
            'video_prompts': video_prompts,
        })
        self._save_session_log()

        self._print_summary_music()
        return self.results

    # ═══════════════════════════════════════
    # MODO 2: PIPELINE COMPLETO
    # Audio + ensamblaje + YouTube
    # ═══════════════════════════════════════
    def run_full(self, duration_hours=None, category=None, num_segments=None,
                 dry_run=False, audio_path=None):
        """
        Pipeline completo: genera audio, ensambla con video, sube a YouTube.
        Requiere clips de video en assets/video_loops/
        """
        self._print_banner("FULL PIPELINE" + (" (DRY RUN)" if dry_run else ""))
        
        target_hours = duration_hours or self.config['audio']['default_duration_hours']
        prompt_info = self._select_prompt(category)

        # Fase Audio
        if audio_path:
            final_audio = Path(audio_path)
            logger.info(f"Usando audio existente: {final_audio}")
        else:
            final_audio = self._phase_audio(prompt_info, num_segments, target_hours)

        # Fase Metadata
        metadata, thumb_path = self._phase_metadata(
            prompt_info, target_hours,
            title_family=getattr(self, '_title_family', None),
            description_mode=getattr(self, '_desc_mode', 'long')
        )

        # Fase Ensamblaje
        final_video = self._phase_assemble(final_audio)

        # Fase YouTube
        upload_result = self._phase_youtube(final_video, metadata, thumb_path, dry_run)

        self.results.update({
            'mode': 'full',
            'audio': str(final_audio),
            'video': str(final_video),
            'thumbnail': str(thumb_path),
            'metadata': metadata,
            'upload': upload_result,
        })
        self._save_session_log()

        self._print_summary_full(dry_run)
        return self.results

    # ═══════════════════════════════════════
    # MODO 3: SOLO ENSAMBLAR
    # Usa audio existente + clips de video
    # ═══════════════════════════════════════
    def run_assemble_only(self, audio_path, dry_run=False, category=None,
                          duration_hours=None):
        """
        Solo ensambla: toma audio existente + clips de video → video final → YouTube.
        """
        self._print_banner("ASSEMBLE-ONLY MODE")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

        # Estimar horas del audio
        target_hours = duration_hours or self.config['audio']['default_duration_hours']
        prompt_info = self._select_prompt(category)

        # Metadata
        metadata, thumb_path = self._phase_metadata(
            prompt_info, target_hours,
            title_family=getattr(self, '_title_family', None),
            description_mode=getattr(self, '_desc_mode', 'long')
        )

        # Ensamblar
        final_video = self._phase_assemble(audio_path)

        # YouTube
        upload_result = self._phase_youtube(final_video, metadata, thumb_path, dry_run)

        self.results.update({
            'mode': 'assemble_only',
            'audio': str(audio_path),
            'video': str(final_video),
            'upload': upload_result,
        })
        self._save_session_log()
        return self.results

    # ═══════════════════════════════════════
    # FASES INTERNAS
    # ═══════════════════════════════════════
    def _select_prompt(self, category):
        """Selecciona prompt de audio."""
        from src.audio.generator import AudioGenerator
        gen = AudioGenerator()
        prompt_info = gen.select_prompt(category=category)
        logger.info(f"🎵 Categoría: {prompt_info['category']}, Mood: {prompt_info['mood']}")
        return prompt_info

    def _phase_audio(self, prompt_info, num_segments, target_hours):
        """Fase de generación y procesamiento de audio."""
        logger.info("\n" + "═" * 50)
        logger.info("🎵 FASE: GENERACIÓN DE AUDIO")
        logger.info("═" * 50)

        # Calcular segmentos necesarios (max 15 — el extender los loopea)
        if num_segments is None:
            seg_sec = self.config['audio']['segment_duration_sec']
            num_segments = min(15, max(3, int(target_hours * 3600 / seg_sec) + 1))

        # 1. Generar segmentos
        logger.info(f"[1/4] Generando {num_segments} segmentos...")
        from src.audio.generator import AudioGenerator
        gen = AudioGenerator()
        batch = gen.generate_batch(num_segments, category=prompt_info['category'], 
                                    prefix=self.session_id)
        segment_paths = [r['path'] for r in batch]

        # 2. Texturas
        logger.info("[2/4] Preparando texturas ambientales...")
        from src.audio.textures import TextureGenerator
        tex_gen = TextureGenerator(self.config)
        texture_map = tex_gen.generate_library(self.prompts_data)
        active = self.config['audio'].get('textures', {}).get('library', [])
        texture_paths = [texture_map[t] for t in active if t in texture_map]

        # 3. Extender
        logger.info(f"[3/4] Extendiendo a {target_hours}h...")
        from src.audio.extender import AudioExtender
        extender = AudioExtender(self.config)
        extended = extender.extend_to_duration(
            segment_paths, target_hours, f"{self.session_id}_extended"
        )

        # 4. Post-procesamiento
        logger.info("[4/4] Post-procesando (LUFS, 432Hz, texturas, limiter)...")
        from src.audio.postprocess import AudioPostProcessor
        pp = AudioPostProcessor(self.config)
        final = pp.process_full_pipeline(
            extended, texture_paths or None, f"{self.session_id}_master"
        )

        logger.info(f"✅ Audio final: {final}")
        self.results['audio'] = str(final)
        return final

    def _phase_metadata(self, prompt_info, hours, title_family=None, description_mode='long'):
        """Genera metadata SEO + thumbnail."""
        logger.info("\n" + "═" * 50)
        logger.info("📝 FASE: METADATA + THUMBNAIL")
        logger.info("═" * 50)

        from src.assembly.metadata import MetadataGenerator
        meta_gen = MetadataGenerator(self.config, self.prompts_data)
        metadata = meta_gen.generate_metadata(
            hours, prompt_info,
            title_family=title_family,
            description_mode=description_mode
        )
        logger.info(f"  Familia: {metadata['title_family']}")
        logger.info(f"  Título: {metadata['title']}")
        logger.info(f"  Label: {metadata['thumbnail_label']}")
        logger.info(f"  Descripción: {'LARGA' if description_mode == 'long' else 'CORTA'}")

        from src.assembly.thumbnail import ThumbnailGenerator
        thumb_gen = ThumbnailGenerator(self.config)
        style = meta_gen._mood_to_style(prompt_info['mood'])
        thumb = thumb_gen.generate(
            hours, prompt_info['mood'], style,
            output_filename=f"{self.session_id}_thumb",
            label=metadata['thumbnail_label']
        )
        
        # Guardar metadata como JSON
        meta_json = Path(self.config['paths']['output']) / f"{self.session_id}_metadata.json"
        with open(meta_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"  Metadata: {meta_json}")

        return metadata, thumb

    def _phase_video_prompts(self, prompt_info):
        """Genera prompts de video para Veo3."""
        logger.info("\n" + "═" * 50)
        logger.info("🎬 FASE: PROMPTS DE VIDEO (PARA VEO3)")
        logger.info("═" * 50)

        from src.assembly.video_prompts import VideoPromptGenerator
        vpg = VideoPromptGenerator(self.config)
        pair = vpg.generate_prompt_pair(prompt_info['mood'])
        formatted = vpg.format_for_user(pair, prompt_info['mood'], prompt_info['category'])

        # Guardar prompts
        prompts_file = Path(self.config['paths']['output']) / f"{self.session_id}_video_prompts.txt"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            f.write(formatted)

        print(formatted)  # Mostrar al usuario
        logger.info(f"  Prompts guardados: {prompts_file}")

        return formatted

    def _phase_assemble(self, audio_path):
        """Ensambla video final (clips Veo3 + audio)."""
        logger.info("\n" + "═" * 50)
        logger.info("📦 FASE: ENSAMBLAJE FINAL")
        logger.info("═" * 50)

        from src.assembly.composer import VideoComposer
        composer = VideoComposer(self.config)
        final = composer.compose(audio_path, f"FINAL_{self.session_id}", add_overlay=True)
        return final

    def _phase_youtube(self, video_path, metadata, thumb_path, dry_run):
        """Sube a YouTube."""
        logger.info("\n" + "═" * 50)
        logger.info(f"📤 FASE: YOUTUBE {'(DRY RUN)' if dry_run else ''}")
        logger.info("═" * 50)

        from src.publish.uploader import YouTubeUploader
        uploader = YouTubeUploader(self.config)

        if dry_run:
            return uploader.upload_dry_run(video_path, metadata, thumb_path)
        else:
            return uploader.upload(video_path, metadata, thumb_path)

    # ═══════════════════════════════════════
    # UTILIDADES
    # ═══════════════════════════════════════
    def _print_banner(self, mode):
        logger.info("=" * 60)
        logger.info(f"🚀 MUSCIAYT PIPELINE — {mode}")
        logger.info(f"   Session: {self.session_id}")
        logger.info("=" * 60)

    def _print_summary_music(self):
        print("\n" + "=" * 50)
        print("📊 SESIÓN COMPLETADA — MUSIC-ONLY")
        print("=" * 50)
        print(f"  🎵 Audio: {self.results.get('audio', 'N/A')}")
        print(f"  🖼️  Thumbnail: {self.results.get('thumbnail', 'N/A')}")
        print(f"\n  📌 SIGUIENTE PASO:")
        print(f"     1. Usa los prompts de video para generar clips en Veo3")
        print(f"     2. Coloca los clips en: assets/video_loops/")
        print(f"     3. Ejecuta: python src/orchestrator.py --assemble-only --audio {self.results.get('audio', '')}")
        print("=" * 50)

    def _print_summary_full(self, dry_run):
        print("\n" + "=" * 50)
        print(f"📊 SESIÓN COMPLETADA — {'DRY RUN' if dry_run else 'PUBLICADO'}")
        print("=" * 50)
        for key, value in self.results.items():
            if key not in ('metadata', 'video_prompts'):
                print(f"  {key}: {value}")
        print("=" * 50)

    def _save_session_log(self):
        log_path = Path(self.config['paths']['logs']) / f"{self.session_id}_results.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--music-only', 'mode', flag_value='music', default=True,
                  help='Solo genera audio + metadata + prompts de video (DEFAULT)')
    @click.option('--full', 'mode', flag_value='full',
                  help='Pipeline completo: audio + video + YouTube')
    @click.option('--assemble-only', 'mode', flag_value='assemble',
                  help='Solo ensamblar audio existente con video')
    @click.option('--dry-run', is_flag=True, help='No subir a YouTube')
    @click.option('--hours', default=None, type=float, help='Duración en horas (2-10)')
    @click.option('--category', default=None, help='Categoría de prompt')
    @click.option('--segments', default=None, type=int, help='Número de segmentos')
    @click.option('--audio', 'audio_path', default=None, type=click.Path(exists=True),
                  help='Path a audio existente (para --assemble-only)')
    @click.option('--title-family', 'title_family', default=None,
                  type=click.Choice(['session', 'emotional', 'identity', 'aesthetic']),
                  help='Familia de título (session/emotional/identity/aesthetic)')
    @click.option('--desc-mode', 'desc_mode', default='long',
                  type=click.Choice(['long', 'short']),
                  help='Modo de descripción (long=completa, short=rápida)')
    def main(mode, dry_run, hours, category, segments, audio_path, title_family, desc_mode):
        """
        MusicaYT — Pipeline de Producción Automatizado
        
        Modos:
          --music-only    Solo genera música + metadata (default)
          --full          Pipeline completo con upload a YouTube
          --assemble-only Solo ensamblar y subir
        
        Títulos:
          --title-family session    -> Deep Focus for Live Trading, NY Session...
          --title-family emotional  -> Stay Calm, No Overtrading, Discipline...
          --title-family identity   -> Focus Like a Funded Trader, Elite...
          --title-family aesthetic   -> Rainy Night, Midnight, 3AM Desk...
        """
        config, prompts_data = load_configs()

        # Validar duración
        if hours:
            min_h = config['audio']['min_duration_hours']
            max_h = config['audio']['max_duration_hours']
            if hours < min_h or hours > max_h:
                print(f"❌ Duración debe estar entre {min_h}h y {max_h}h")
                sys.exit(1)

        setup_logging(config)
        pipeline = Pipeline(config, prompts_data)
        pipeline._title_family = title_family
        pipeline._desc_mode = desc_mode

        try:
            if mode == 'music':
                pipeline.run_music_only(hours, category, segments)
            elif mode == 'full':
                pipeline.run_full(hours, category, segments, dry_run, audio_path)
            elif mode == 'assemble':
                if not audio_path:
                    print("❌ --assemble-only requiere --audio <path>")
                    sys.exit(1)
                pipeline.run_assemble_only(audio_path, dry_run, category, hours)

        except KeyboardInterrupt:
            logger.warning("Interrumpido por el usuario")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Pipeline falló: {e}", exc_info=True)
            sys.exit(1)

    main()

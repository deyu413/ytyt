"""
MusicaYT — Compositor de Video Final
=====================================
Combina los clips de video del usuario (Veo3) con el audio generado.

El usuario coloca sus clips en assets/video_loops/ y el sistema:
1. Detecta los clips disponibles
2. Los loopea/alterna para cubrir la duración del audio
3. Ensambla video + audio con FFmpeg
4. Añade overlay sutil opcional (nombre del canal)
"""

import os
import subprocess
import logging
import math
import random
from pathlib import Path
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


class VideoComposer:
    """Ensambla video final: loops manuales (Veo3) + audio generado."""

    def __init__(self, config):
        self.config = config
        self.video_config = config.get('video', {})
        self.loops_dir = Path(config['paths']['video_loops'])
        self.output_dir = Path(config['paths']['output'])
        self.loops_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_video_loops(self):
        """Detecta clips de video disponibles en assets/video_loops/."""
        extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
        clips = []
        for f in sorted(self.loops_dir.iterdir()):
            if f.suffix.lower() in extensions and f.is_file():
                clips.append(f)

        if not clips:
            logger.error(f"❌ No se encontraron clips de video en: {self.loops_dir}")
            logger.error(f"   Coloca tus clips de Veo3 allí (loop_1.mp4, loop_2.mp4)")
            raise FileNotFoundError(
                f"No hay clips de video en {self.loops_dir}. "
                f"Genera tus clips con Veo3 y colócalos allí."
            )

        for clip in clips:
            dur = self._get_duration(clip)
            size = clip.stat().st_size / (1024 * 1024)
            logger.info(f"  Clip encontrado: {clip.name} ({dur:.1f}s, {size:.1f}MB)")

        return clips

    def compose(self, audio_path, output_filename=None, add_overlay=False):
        """
        Compone el video final.
        
        Args:
            audio_path: Path al audio masterizado (2-10h)
            output_filename: Nombre del archivo final
            add_overlay: Añadir texto del canal en esquina
            
        Returns:
            Path al video final
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"FINAL_{timestamp}"

        output_path = self.output_dir / f"{output_filename}.mp4"

        # Detectar clips de video
        clips = self.find_video_loops()
        audio_duration = self._get_duration(audio_path)

        logger.info(f"Composición:")
        logger.info(f"  Audio: {audio_duration/3600:.2f}h ({audio_path.name})")
        logger.info(f"  Clips: {len(clips)} disponibles")

        if len(clips) == 1:
            # Un solo clip: loop directo
            video_source = self._prepare_single_loop(clips[0], audio_duration)
        else:
            # Múltiples clips: concatenar y alternar
            video_source = self._prepare_multi_loop(clips, audio_duration)

        # Ensamblar video + audio
        if add_overlay:
            result = self._compose_with_overlay(video_source, audio_path, output_path, audio_duration)
        else:
            result = self._compose_simple(video_source, audio_path, output_path, audio_duration)

        # Limpiar archivo temporal de concatenación si existe
        concat_temp = self.output_dir / "_temp_concat.mp4"
        if concat_temp.exists():
            concat_temp.unlink()

        if result.exists():
            size_gb = result.stat().st_size / (1024 ** 3)
            logger.info(f"✅ Video final: {result} ({size_gb:.2f} GB)")
            return result
        else:
            raise RuntimeError("Composición falló")

    def _prepare_single_loop(self, clip_path, target_duration):
        """Un solo clip: se usa stream_loop de FFmpeg directamente."""
        logger.info(f"  Modo: loop infinito de {clip_path.name}")
        return clip_path  # FFmpeg hará el loop con -stream_loop

    def _prepare_multi_loop(self, clips, target_duration):
        """
        Múltiples clips: crea un archivo concat que alterna entre ellos.
        Calcula cuántas repeticiones se necesitan para cubrir la duración.
        """
        total_clip_duration = sum(self._get_duration(c) for c in clips)
        repetitions = math.ceil(target_duration / total_clip_duration) + 1

        # Crear archivo de lista para FFmpeg concat
        concat_list = self.output_dir / "_concat_list.txt"
        with open(concat_list, 'w', encoding='utf-8') as f:
            for _ in range(repetitions):
                for clip in clips:
                    # FFmpeg concat requiere paths con / y escapados
                    safe_path = str(clip.absolute()).replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

        logger.info(f"  Modo: alternancia de {len(clips)} clips × {repetitions} repeticiones")

        # Concatenar en un archivo temporal
        concat_temp = self.output_dir / "_temp_concat.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            '-t', str(target_duration),
            str(concat_temp)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            concat_list.unlink(missing_ok=True)  # Limpiar lista

            if result.returncode == 0 and concat_temp.exists():
                return concat_temp
            else:
                logger.warning(f"Concat falló, usando primer clip en loop: {result.stderr[:200]}")
                concat_list.unlink(missing_ok=True)
                return clips[0]

        except subprocess.TimeoutExpired:
            concat_list.unlink(missing_ok=True)
            logger.warning("Concat timeout, usando primer clip")
            return clips[0]

    def _compose_simple(self, video_path, audio_path, output_path, audio_duration):
        """Composición: video + audio, sin overlay."""
        is_single_loop = not str(video_path).endswith('_temp_concat.mp4')

        cmd = ['ffmpeg', '-y']

        if is_single_loop:
            cmd.extend(['-stream_loop', '-1'])

        cmd.extend([
            '-i', str(video_path),
            '-i', str(audio_path),
            '-t', str(audio_duration),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'h264_nvenc',
            '-preset', 'slow',
            '-cq', str(self.video_config.get('crf', 20)),
            '-c:a', 'aac', '-b:a', '320k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-movflags', '+faststart',
            str(output_path)
        ])

        logger.info("Componiendo video final (puede tomar horas para 4h+ de contenido)...")
        return self._execute_ffmpeg(cmd, output_path, audio_duration)

    def _compose_with_overlay(self, video_path, audio_path, output_path, audio_duration):
        """Composición con overlay del nombre del canal."""
        channel = self.config.get('metadata', {}).get('channel_name', 'Protocol Systems')

        overlay_filter = (
            f"drawtext=text='{channel}':"
            f"fontcolor=white@0.2:"
            f"fontsize=14:"
            f"x=w-tw-20:y=h-th-15:"
            f"font='Consolas'"
        )

        is_single = not str(video_path).endswith('_temp_concat.mp4')
        cmd = ['ffmpeg', '-y']
        if is_single:
            cmd.extend(['-stream_loop', '-1'])

        cmd.extend([
            '-i', str(video_path),
            '-i', str(audio_path),
            '-t', str(audio_duration),
            '-filter_complex', f"[0:v:0]{overlay_filter}[vout]",
            '-map', '[vout]', '-map', '1:a:0',
            '-c:v', self.video_config.get('codec', 'libx264'),
            '-crf', str(self.video_config.get('crf', 20)),
            '-preset', self.video_config.get('preset', 'slow'),
            '-c:a', 'aac', '-b:a', '320k',
            '-pix_fmt', 'yuv420p',
            '-shortest', '-movflags', '+faststart',
            str(output_path)
        ])

        logger.info("Componiendo video con overlay...")
        return self._execute_ffmpeg(cmd, output_path, audio_duration)

    def _execute_ffmpeg(self, cmd, output_path, expected_duration):
        """Ejecuta FFmpeg con timeout generoso."""
        timeout = int(expected_duration * 0.5) + 7200  # Generoso para videos largos

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                logger.error(f"FFmpeg: {result.stderr[-500:]}")
                raise RuntimeError(f"FFmpeg code {result.returncode}")
            return output_path
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout ({timeout}s)")
            raise

    def _get_duration(self, file_path):
        """Obtiene duración de un archivo de audio/video."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            return 14400  # Default 4h


# --- CLI ---
if __name__ == "__main__":
    import click

    @click.command()
    @click.argument('audio', type=click.Path(exists=True))
    @click.option('--output', default=None, help='Nombre del video final')
    @click.option('--overlay', is_flag=True, help='Añadir overlay del canal')
    @click.option('--list-clips', is_flag=True, help='Listar clips disponibles')
    def main(audio, output, overlay, list_clips):
        """Compone video final (clips Veo3 + audio generado)."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        composer = VideoComposer(config)

        if list_clips:
            try:
                clips = composer.find_video_loops()
                print(f"\n📹 {len(clips)} clips disponibles:")
                for c in clips:
                    print(f"  • {c.name}")
            except FileNotFoundError as e:
                print(f"\n{e}")
            return

        result = composer.compose(audio, output, add_overlay=overlay)
        print(f"✓ Video final: {result}")

    main()

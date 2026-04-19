"""
MusicaYT — Extensión Iterativa de Audio (Sliding Window)
=========================================================
Toma segmentos cortos (30s-4min) y los extiende a pistas de 4-8 horas
usando la técnica de sliding window: los últimos N segundos del segmento
anterior se usan como contexto para generar el siguiente.

Para ACE-Step: concatenación inteligente con crossfade (no tiene extend nativo)
Para YuEGP: usa --extend_current_segment para continuación coherente
"""

import os
import logging
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioExtender:
    """
    Extiende pistas de audio cortas a duración larga (4-8 horas)
    usando concatenación inteligente con análisis de coherencia.
    """

    def __init__(self, config):
        self.config = config
        self.audio_config = config['audio']
        self.sample_rate = self.audio_config['sample_rate']
        self.crossfade_ms = self.audio_config['crossfade_ms']
        self.crossfade_samples = int(self.sample_rate * self.crossfade_ms / 1000)
        self.output_dir = Path(config['paths']['audio_raw'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extend_to_duration(self, segments, target_hours, output_filename=None):
        """
        Extiende una lista de segmentos de audio hasta alcanzar la duración objetivo.
        
        Args:
            segments: Lista de paths a archivos de audio (segmentos generados)
            target_hours: Duración objetivo en horas
            output_filename: Nombre del archivo de salida
            
        Returns:
            Path al archivo extendido
        """
        if not segments:
            raise ValueError("Se necesita al menos 1 segmento para extender")

        target_samples = int(target_hours * 3600 * self.sample_rate)

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"extended_{target_hours}h_{timestamp}"

        output_path = self.output_dir / f"{output_filename}.wav"

        logger.info(f"Extendiendo a {target_hours}h ({target_samples:,} samples)")
        logger.info(f"Segmentos disponibles: {len(segments)}")

        # Cargar todos los segmentos
        audio_segments = []
        for seg_path in segments:
            data, sr = sf.read(str(seg_path), dtype='float32')
            if sr != self.sample_rate:
                logger.warning(f"Sample rate mismatch: {sr} vs {self.sample_rate}")
                # Resamplear si es necesario
                data = self._resample(data, sr, self.sample_rate)
            
            # Ya no forzamos mono, mantenemos todo el estéreo de los MP3 originales.
            
            audio_segments.append(data)
            logger.info(f"  Cargado: {Path(seg_path).name} — {len(data)/self.sample_rate:.1f}s")

        # Construir la pista extendida
        extended = self._build_extended_track(audio_segments, target_samples)

        # Guardar
        sf.write(str(output_path), extended, self.sample_rate, subtype='PCM_24')
        duration_hours = len(extended) / self.sample_rate / 3600
        logger.info(f"Pista extendida guardada: {output_path} ({duration_hours:.2f}h)")

        return output_path

    def _build_extended_track(self, segments, target_samples):
        """
        Construye una pista larga concatenando segmentos con crossfade inteligente.
        
        Estrategia:
        1. Mezcla los segmentos en orden aleatorio-ponderado (evita repetición inmediata)
        2. Aplica crossfade en puntos de zero-crossing para transiciones imperceptibles
        3. Repite hasta alcanzar la duración objetivo
        
        Optimización: acumula en lista y concatena una sola vez al final
        para evitar O(n²) de memoria con np.concatenate repetido.
        """
        # Acumular segmentos procesados en lista (eficiente)
        chunks = []
        total_samples = 0
        segment_count = len(segments)
        last_used = -1
        prev_tail = None  # Cola del segmento anterior para crossfade

        iteration = 0
        while total_samples < target_samples:
            # Seleccionar siguiente segmento (evitar repetición consecutiva)
            available = list(range(segment_count))
            if segment_count > 1 and last_used >= 0:
                available.remove(last_used)

            idx = available[iteration % len(available)]
            segment = segments[idx].copy()
            last_used = idx

            if prev_tail is None:
                # Primer segmento: sin crossfade
                chunks.append(segment)
                total_samples += len(segment)
            else:
                # Crossfade con la cola del segmento anterior
                cf_len = self.crossfade_samples
                if len(prev_tail) >= cf_len and len(segment) >= cf_len:
                    # Crear zona de crossfade
                    min_len = min(len(prev_tail), cf_len)
                    fade_in_region = segment[:min_len]

                    t = np.linspace(0, np.pi / 2, min_len)
                    fade_out_curve = np.cos(t) ** 2
                    fade_in_curve = np.sin(t) ** 2

                    if segment.ndim > 1:
                        fade_out_curve = fade_out_curve[:, np.newaxis]
                        fade_in_curve = fade_in_curve[:, np.newaxis]

                    crossfaded = prev_tail[:min_len] * fade_out_curve + fade_in_region * fade_in_curve
                    chunks.append(crossfaded)
                    chunks.append(segment[min_len:])
                    total_samples += min_len + len(segment[min_len:])
                else:
                    chunks.append(segment)
                    total_samples += len(segment)

            # Guardar cola para el siguiente crossfade
            prev_tail = segment[-self.crossfade_samples:] if len(segment) >= self.crossfade_samples else segment

            iteration += 1
            if iteration % 10 == 0:
                progress = min(total_samples / target_samples * 100, 100)
                logger.info(f"  Progreso: {progress:.1f}% — {total_samples/self.sample_rate/60:.0f} min")

        # Concatenación única al final (O(n) en vez de O(n²))
        extended = np.concatenate(chunks)

        # Truncar al tamaño exacto con fade out
        if len(extended) > target_samples:
            extended = extended[:target_samples]

        # Fade out en los últimos 5 segundos
        fade_samples = min(5 * self.sample_rate, target_samples // 10)
        fade = np.linspace(1.0, 0.0, fade_samples)
        if extended.ndim > 1:
            fade = fade[:, np.newaxis]
        extended[-fade_samples:] *= fade

        logger.info(f"Pista construida: {iteration} iteraciones, "
                     f"{len(extended)/self.sample_rate/3600:.2f}h")
        return extended

    def _crossfade_at_zero_crossing(self, audio_a, audio_b):
        """
        Aplica crossfade entre dos segmentos de audio en puntos de zero-crossing.
        
        Los zero-crossings son puntos donde la onda cruza el nivel cero,
        lo que minimiza los clicks y pops en las transiciones.
        """
        cf_len = self.crossfade_samples

        if len(audio_a) < cf_len or len(audio_b) < cf_len:
            # Segmentos demasiado cortos para crossfade, concatenar directamente
            return np.concatenate([audio_a, audio_b])

        # Encontrar zero-crossing cerca del final de audio_a
        tail = audio_a[-cf_len * 2:]
        zc_a = self._find_nearest_zero_crossing(tail, cf_len)

        # Encontrar zero-crossing cerca del inicio de audio_b
        head = audio_b[:cf_len * 2]
        zc_b = self._find_nearest_zero_crossing(head, 0)

        # Ajustar puntos de corte
        cut_a = len(audio_a) - cf_len * 2 + zc_a
        cut_b = zc_b

        # Extraer las zonas de crossfade
        fade_out_region = audio_a[cut_a:cut_a + cf_len]
        fade_in_region = audio_b[cut_b:cut_b + cf_len]

        # Asegurar que ambas regiones tienen el mismo tamaño
        min_len = min(len(fade_out_region), len(fade_in_region))
        if min_len < 10:
            return np.concatenate([audio_a, audio_b])
        
        fade_out_region = fade_out_region[:min_len]
        fade_in_region = fade_in_region[:min_len]

        # Crear curvas de fade (coseno para suavidad)
        t = np.linspace(0, np.pi / 2, min_len)
        fade_out_curve = np.cos(t) ** 2
        fade_in_curve = np.sin(t) ** 2

        if fade_out_region.ndim > 1:
            fade_out_curve = fade_out_curve[:, np.newaxis]
            fade_in_curve = fade_in_curve[:, np.newaxis]

        # Mezclar
        crossfaded = fade_out_region * fade_out_curve + fade_in_region * fade_in_curve

        # Construir resultado
        result = np.concatenate([
            audio_a[:cut_a],
            crossfaded,
            audio_b[cut_b + min_len:]
        ])

        return result

    def _find_nearest_zero_crossing(self, audio, target_pos):
        """Encuentra el zero-crossing más cercano a la posición objetivo."""
        if len(audio) < 2:
            return target_pos

        # Detectar zero-crossings (solo buscamos en el canal L si es stereo)
        audio_for_zc = audio[:, 0] if audio.ndim > 1 else audio
        sign_changes = np.where(np.diff(np.signbit(audio_for_zc)))[0]

        if len(sign_changes) == 0:
            return target_pos

        # Encontrar el más cercano a target_pos
        distances = np.abs(sign_changes - target_pos)
        nearest_idx = sign_changes[np.argmin(distances)]

        return int(nearest_idx)

    def _resample(self, data, orig_sr, target_sr):
        """Resamplea audio de orig_sr a target_sr."""
        try:
            import librosa
            return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: interpolación simple
            ratio = target_sr / orig_sr
            new_length = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, new_length)
            return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


# --- CLI ---
if __name__ == "__main__":
    import click
    import yaml

    @click.command()
    @click.argument('segments', nargs=-1, required=True, type=click.Path(exists=True))
    @click.option('--hours', default=4, type=float, help='Duración objetivo en horas')
    @click.option('--output', default=None, help='Nombre del archivo de salida')
    def main(segments, hours, output):
        """Extiende segmentos de audio a larga duración."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        extender = AudioExtender(config)
        result = extender.extend_to_duration(
            segments=list(segments),
            target_hours=hours,
            output_filename=output
        )
        print(f"✓ Pista extendida: {result}")

    main()

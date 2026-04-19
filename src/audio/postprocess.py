"""
MusicaYT — Post-Procesamiento de Audio
=======================================
Pipeline automático de masterización:
1. Normalización a -14 LUFS (estándar YouTube)
2. Crossfade inteligente en zero-crossings
3. Retuning de 440Hz → 432Hz
4. Mezcla de capas: pista principal + texturas ambientales
5. Export: WAV 24bit / MP3 320kbps
"""

import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPostProcessor:
    """Pipeline de post-procesamiento y masterización de audio."""

    def __init__(self, config):
        self.config = config
        self.audio_config = config['audio']
        self.sample_rate = self.audio_config['sample_rate']
        self.target_lufs = self.audio_config['target_lufs']
        self.output_dir = Path(config['paths']['audio_final'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_full_pipeline(self, main_audio_path, texture_paths=None, output_filename=None):
        """
        Ejecuta el pipeline completo de post-procesamiento.
        
        Args:
            main_audio_path: Path a la pista de audio principal (ya extendida)
            texture_paths: Lista de paths a texturas para mezclar (opcional)
            output_filename: Nombre del archivo final
            
        Returns:
            Path al archivo masterizado
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"master_{timestamp}"

        logger.info("=" * 60)
        logger.info("INICIO DEL PIPELINE DE POST-PROCESAMIENTO")
        logger.info("=" * 60)

        # 1. Cargar audio principal
        logger.info("[1/5] Cargando audio principal...")
        audio, sr = sf.read(str(main_audio_path), dtype='float32')
        logger.info(f"  Duración: {len(audio)/sr/3600:.2f}h, SR: {sr}, Canales: {'Stereo' if audio.ndim > 1 else 'Mono'}")

        # 2. Retuning 432Hz (si está habilitado)
        if self.audio_config.get('retune_432hz', False):
            logger.info("[2/5] Retuning a 432Hz...")
            audio = self._retune_432hz(audio, sr)
        else:
            logger.info("[2/5] Retuning 432Hz deshabilitado, saltando...")

        # 3. Mezclar texturas
        if texture_paths:
            logger.info(f"[3/5] Mezclando {len(texture_paths)} texturas...")
            mix_level_db = self.audio_config.get('textures', {}).get('mix_level_db', -12)
            audio = self._mix_textures(audio, texture_paths, sr, mix_level_db)
        else:
            logger.info("[3/5] Sin texturas para mezclar, saltando...")

        # 4. Normalización LUFS
        logger.info(f"[4/5] Normalizando a {self.target_lufs} LUFS...")
        audio = self._normalize_lufs(audio, sr)

        # 5. Aplicar limitador suave (evita clipping)
        logger.info("[5/5] Aplicando limitador y guardando...")
        audio = self._soft_limiter(audio, threshold=0.95)

        # Guardar WAV
        wav_path = self.output_dir / f"{output_filename}.wav"
        sf.write(str(wav_path), audio, sr, subtype='PCM_24')
        logger.info(f"  WAV guardado: {wav_path}")

        # Opcional: generar MP3 también
        mp3_path = self._export_mp3(wav_path)
        if mp3_path:
            logger.info(f"  MP3 guardado: {mp3_path}")

        logger.info("=" * 60)
        logger.info(f"POST-PROCESAMIENTO COMPLETADO: {wav_path}")
        logger.info("=" * 60)

        return wav_path

    def _retune_432hz(self, audio, sr):
        """
        Retune de 440Hz standard a 432Hz.
        
        Ratio: 432/440 = 0.981818...
        Procesa en chunks de 5 minutos para mantener el uso de RAM
        controlado en pistas de 4-10 horas.
        """
        ratio = 432.0 / 440.0  # = 0.98182...

        try:
            import librosa
            
            # Para audios largos (>10 min), procesar en chunks
            chunk_duration = 5 * 60  # 5 minutos
            chunk_samples = chunk_duration * sr
            overlap_samples = sr  # 1 segundo de overlap para suavizar uniones
            
            if len(audio) <= chunk_samples * 2:
                # Audio corto: procesar de una vez
                t_audio = audio.T if audio.ndim > 1 else audio
                t_retuned = librosa.effects.pitch_shift(
                    t_audio, sr=sr,
                    n_steps=-0.3176,
                    res_type='kaiser_best'
                )
                audio_retuned = t_retuned.T if audio.ndim > 1 else t_retuned
            else:
                # Audio largo: procesar por chunks
                chunks = []
                pos = 0
                total_chunks = (len(audio) // chunk_samples) + 1
                chunk_idx = 0
                
                while pos < len(audio):
                    end = min(pos + chunk_samples + overlap_samples, len(audio))
                    chunk = audio[pos:end]
                    
                    t_chunk = chunk.T if chunk.ndim > 1 else chunk
                    t_retuned_chunk = librosa.effects.pitch_shift(
                        t_chunk, sr=sr,
                        n_steps=-0.3176,
                        res_type='kaiser_fast'  # Más rápido para chunks
                    )
                    retuned_chunk = t_retuned_chunk.T if chunk.ndim > 1 else t_retuned_chunk
                    
                    if chunks and overlap_samples > 0 and len(retuned_chunk) > overlap_samples:
                        # Crossfade en el overlap
                        fade = np.linspace(0, 1, overlap_samples)
                        if retuned_chunk.ndim > 1:
                            fade = fade[:, np.newaxis]
                        retuned_chunk[:overlap_samples] = (
                            chunks[-1][-overlap_samples:] * (1 - fade) + 
                            retuned_chunk[:overlap_samples] * fade
                        )
                        chunks[-1] = chunks[-1][:-overlap_samples]
                    
                    chunks.append(retuned_chunk)
                    pos += chunk_samples
                    chunk_idx += 1
                    
                    if chunk_idx % 5 == 0:
                        logger.info(f"  Retuning: chunk {chunk_idx}/{total_chunks}")
                
                audio_retuned = np.concatenate(chunks)
            
            logger.info(f"  Retuning completado con librosa (pitch shift -0.32 semitonos)")
            return audio_retuned

        except ImportError:
            # Fallback: resampleo manual (cambia ligeramente la duración)
            logger.warning("  librosa no disponible, usando resampleo manual (duración afectada ~2%)")
            new_length = int(len(audio) / ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            retuned = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
            return retuned

    def _mix_textures(self, main_audio, texture_paths, sr, mix_level_db=-12):
        """
        Mezcla texturas ambientales con la pista principal.
        Las texturas se repiten en loop para cubrir toda la duración.
        
        Args:
            main_audio: Array numpy de audio principal
            texture_paths: Lista de paths a archivos de textura
            sr: Sample rate
            mix_level_db: Nivel de las texturas respecto al principal (en dB)
        """
        target_length = len(main_audio)
        mix_factor = 10 ** (mix_level_db / 20.0)  # dB a linear

        for tex_path in texture_paths:
            try:
                tex_audio, tex_sr = sf.read(str(tex_path), dtype='float32')

                # Adaptar canales para coincidir con main_audio
                if main_audio.ndim > 1 and tex_audio.ndim == 1:
                    tex_audio = np.tile(tex_audio[:, np.newaxis], (1, main_audio.shape[1]))
                elif main_audio.ndim == 1 and tex_audio.ndim > 1:
                    tex_audio = np.mean(tex_audio, axis=1)

                # Resamplear si es necesario
                if tex_sr != sr:
                    try:
                        import librosa
                        t_tex = tex_audio.T if tex_audio.ndim > 1 else tex_audio
                        t_res = librosa.resample(t_tex, orig_sr=tex_sr, target_sr=sr)
                        tex_audio = t_res.T if tex_audio.ndim > 1 else t_res
                    except ImportError:
                        ratio = sr / tex_sr
                        new_len = int(len(tex_audio) * ratio)
                        if tex_audio.ndim > 1:
                            res_channels = []
                            indices = np.linspace(0, len(tex_audio) - 1, new_len)
                            for ch in range(tex_audio.shape[1]):
                                res_channels.append(np.interp(indices, np.arange(len(tex_audio)), tex_audio[:, ch]))
                            tex_audio = np.column_stack(res_channels).astype(np.float32)
                        else:
                            indices = np.linspace(0, len(tex_audio) - 1, new_len)
                            tex_audio = np.interp(indices, np.arange(len(tex_audio)), tex_audio).astype(np.float32)

                # Repetir textura en loop hasta cubrir la duración
                if len(tex_audio) < target_length:
                    repeats = (target_length // len(tex_audio)) + 1
                    if tex_audio.ndim > 1:
                        tex_audio = np.tile(tex_audio.T, repeats).T
                    else:
                        tex_audio = np.tile(tex_audio, repeats)
                tex_audio = tex_audio[:target_length]

                # Mezclar con el nivel configurado
                main_audio = main_audio + tex_audio * mix_factor
                logger.info(f"  Textura mezclada: {Path(tex_path).name} @ {mix_level_db}dB")

            except Exception as e:
                logger.error(f"  Error mezclando textura {tex_path}: {e}")
                continue

        return main_audio

    def _normalize_lufs(self, audio, sr):
        """
        Normaliza el audio al nivel LUFS objetivo.
        
        YouTube recomienda -14 LUFS para evitar que el audio sea
        comprimido adicionalmente por su normalizador interno.
        """
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(audio)

            if np.isinf(current_lufs) or np.isnan(current_lufs):
                logger.warning("  LUFS inválido (audio posiblemente en silencio)")
                return audio

            logger.info(f"  LUFS actual: {current_lufs:.1f}, objetivo: {self.target_lufs}")
            audio_normalized = pyln.normalize.loudness(audio, current_lufs, self.target_lufs)
            return audio_normalized

        except ImportError:
            # Fallback: normalización RMS simple
            logger.warning("  pyloudnorm no disponible, usando normalización RMS")
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                # -14 LUFS ≈ -14 dBFS RMS (aproximación burda)
                target_rms = 10 ** (self.target_lufs / 20.0)
                audio = audio * (target_rms / rms)
            return audio

    def _soft_limiter(self, audio, threshold=0.95):
        """
        Limitador suave para evitar clipping sin distorsión audible.
        Usa compresión suave (tanh) en lugar de hard clipping.
        """
        # Normalizar picos si superan el threshold
        peak = np.max(np.abs(audio))
        if peak > threshold:
            # Soft clipping con tanh
            audio = np.tanh(audio / threshold) * threshold
            logger.info(f"  Limitador aplicado (peak original: {peak:.3f})")
        return audio

    def _export_mp3(self, wav_path, bitrate='320k'):
        """Convierte WAV a MP3 usando FFmpeg."""
        mp3_path = wav_path.with_suffix('.mp3')

        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(wav_path),
                '-codec:a', 'libmp3lame',
                '-b:a', bitrate,
                '-q:a', '0',
                str(mp3_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                return mp3_path
            else:
                logger.warning(f"  FFmpeg MP3 export falló: {result.stderr[:200]}")
                return None

        except FileNotFoundError:
            logger.warning("  FFmpeg no encontrado — MP3 no generado")
            return None
        except subprocess.TimeoutExpired:
            logger.error("  FFmpeg timeout en conversión MP3")
            return None


# --- CLI ---
if __name__ == "__main__":
    import click
    import yaml

    @click.command()
    @click.argument('audio_path', type=click.Path(exists=True))
    @click.option('--textures', '-t', multiple=True, type=click.Path(exists=True),
                  help='Paths a texturas para mezclar')
    @click.option('--output', '-o', default=None, help='Nombre del archivo de salida')
    @click.option('--no-432hz', is_flag=True, help='Deshabilitar retuning 432Hz')
    @click.option('--test', is_flag=True, help='Modo test: verifica el pipeline con un tono')
    def main(audio_path, textures, output, no_432hz, test):
        """Post-procesa audio para MusicaYT."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if no_432hz:
            config['audio']['retune_432hz'] = False

        if test:
            # Generar un tono de prueba de 10 segundos
            print("🧪 Modo test: generando tono de prueba...")
            sr = config['audio']['sample_rate']
            duration = 10
            t = np.linspace(0, duration, sr * duration, dtype=np.float32)
            test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz, A4
            test_path = Path("assets/audio_raw/test_tone.wav")
            test_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(test_path), test_audio, sr)
            audio_path = str(test_path)
            output = "test_output"
            print(f"  Tono de prueba generado: {test_path}")

        pp = AudioPostProcessor(config)
        result = pp.process_full_pipeline(
            main_audio_path=audio_path,
            texture_paths=list(textures) if textures else None,
            output_filename=output
        )
        print(f"\n✓ Audio masterizado: {result}")

    main()

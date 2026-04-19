"""
MusicaYT — Generador de Thumbnails
====================================
Genera thumbnails estilo "Functional Luxury" para YouTube:
- Fondo oscuro (#121212) con acentos neon
- Tipografía monoespaciada (JetBrains Mono / Consolas)
- Elementos: duración, frecuencia, estilo
- Resolución: 1280x720px (estándar YouTube)
"""

import os
import logging
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """Genera thumbnails premium para el nicho de trading."""

    def __init__(self, config):
        self.config = config
        self.thumb_config = config.get('thumbnail', {})
        self.width = self.thumb_config.get('width', 1280)
        self.height = self.thumb_config.get('height', 720)
        self.bg_color = self.thumb_config.get('bg_color', '#121212')
        self.accent = self.thumb_config.get('accent_color', '#007AFF')
        self.secondary = self.thumb_config.get('secondary_color', '#FF3B30')
        self.text_color = self.thumb_config.get('text_color', '#FFFFFF')
        self.font_family = self.thumb_config.get('font_family', 'JetBrains Mono')
        self.font_fallback = self.thumb_config.get('font_fallback', 'Consolas')
        self.output_dir = Path(config['paths'].get('thumbnails_out', 'output/thumbnails'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, duration_hours, mood, style, video_frame_path=None,
                 output_filename=None, label=None):
        """
        Genera un thumbnail minimalista y consistente.
        
        Sistema de diseño:
        - Misma estructura siempre (identidad de marca)
        - Fondo: frame del video (oscurecido) o gradiente oscuro
        - Label central: 1-2 palabras máximo (LIVE FOCUS, STAY CALM, etc.)
        - Sin subtítulos, sin badges, sin ruido visual
        
        Args:
            duration_hours: Duración del video
            mood: Estado de ánimo
            style: Estilo visual del audio
            video_frame_path: Frame del video para fondo (recomendado)
            output_filename: Nombre del archivo de salida
            label: Texto corto para el thumbnail (1-2 palabras, ej: 'LIVE FOCUS')
            
        Returns:
            Path al thumbnail generado
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"thumb_{timestamp}"

        output_path = self.output_dir / f"{output_filename}.png"

        # Crear imagen base
        if video_frame_path and Path(video_frame_path).exists():
            img = self._create_from_frame(video_frame_path)
        else:
            img = self._create_gradient_background()

        draw = ImageDraw.Draw(img)

        # Cargar fuentes
        font_label = self._load_font(90)
        font_duration = self._load_font(28)
        font_watermark = self._load_font(16)

        # --- Layout Minimalista ---
        # Solo 3 elementos: label central + duración sutil + watermark

        # 1. Label central (1-2 palabras, el alma del thumbnail)
        label_text = (label or 'DEEP FOCUS').upper()
        label_bbox = draw.textbbox((0, 0), label_text, font=font_label)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        label_x = (self.width - label_w) // 2
        label_y = (self.height - label_h) // 2

        self._draw_text_with_glow(
            draw, (label_x, label_y), label_text,
            font_label, self.text_color, glow_color=self.accent
        )

        # 2. Duración sutil (esquina inferior izquierda, discreto)
        duration_text = f"{int(duration_hours)}H · 432Hz"
        draw.text(
            (40, self.height - 50),
            duration_text, fill=(255, 255, 255, 100), font=font_duration
        )

        # 3. Watermark del canal (esquina inferior derecha, muy sutil)
        channel = self.config.get('metadata', {}).get('channel_name', '')
        if channel:
            ch_bbox = draw.textbbox((0, 0), channel, font=font_watermark)
            ch_w = ch_bbox[2] - ch_bbox[0]
            draw.text(
                (self.width - ch_w - 30, self.height - 40),
                channel, fill=(255, 255, 255, 60), font=font_watermark
            )

        # Guardar
        img.save(str(output_path), 'PNG', quality=95)
        logger.info(f"Thumbnail generado: {output_path} — label: '{label_text}'")
        return output_path


    def _create_gradient_background(self):
        """Crea un fondo con gradiente oscuro."""
        img = Image.new('RGBA', (self.width, self.height))
        draw = ImageDraw.Draw(img)

        # Gradiente vertical oscuro
        bg = self._hex_to_rgb(self.bg_color)
        for y in range(self.height):
            ratio = y / self.height
            r = int(bg[0] * (1 - ratio * 0.5))
            g = int(bg[1] * (1 - ratio * 0.5))
            b = int(bg[2] * (1 - ratio * 0.3))
            draw.line([(0, y), (self.width, y)], fill=(r, g, b, 255))

        # Añadir elementos decorativos sutiles
        accent_rgb = self._hex_to_rgb(self.accent)
        
        # Líneas horizontales tenues (estilo terminal)
        for y in range(0, self.height, 4):
            draw.line(
                [(0, y), (self.width, y)],
                fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 8),
                width=1
            )

        return img

    def _create_from_frame(self, frame_path):
        """Crea fondo desde un frame de video (borroso + darkened)."""
        img = Image.open(frame_path).convert('RGBA')
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)

        # Aplicar blur y oscurecer
        img = img.filter(ImageFilter.GaussianBlur(radius=15))

        # Overlay oscuro
        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 180))
        img = Image.alpha_composite(img, overlay)

        return img

    def _draw_text_with_glow(self, draw, pos, text, font, fill, glow_color=None):
        """Dibuja texto con efecto de resplandor sutil."""
        if glow_color:
            glow_rgb = self._hex_to_rgb(glow_color)
            # Dibujar texto desplazado para efecto glow
            for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (0, 2)]:
                draw.text(
                    (pos[0] + offset[0], pos[1] + offset[1]),
                    text,
                    fill=(glow_rgb[0], glow_rgb[1], glow_rgb[2], 40),
                    font=font
                )

        draw.text(pos, text, fill=fill, font=font)

    def _load_font(self, size):
        """Carga la fuente monoespaciada, con fallbacks."""
        font_names = [
            # Paths directos de Windows
            'C:/Windows/Fonts/JetBrainsMono-Regular.ttf',
            'C:/Windows/Fonts/consola.ttf',  # Consolas
            'C:/Windows/Fonts/consolab.ttf',  # Consolas Bold
            'C:/Windows/Fonts/cour.ttf',      # Courier New
            # Nombres genéricos (Linux/Mac)
            self.font_family,
            self.font_fallback,
            'Consolas',
            'Courier New',
        ]

        for name in font_names:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                try:
                    return ImageFont.truetype(f"{name}.ttf", size)
                except (OSError, IOError):
                    continue

        # Fallback absoluto: fuente default de Pillow
        logger.warning(f"No se encontró fuente monoespaciada, usando default")
        return ImageFont.load_default()

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convierte color hex a tupla RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# --- CLI ---
if __name__ == "__main__":
    import click
    import yaml

    @click.command()
    @click.option('--hours', default=4, type=float, help='Duración en horas')
    @click.option('--mood', default='calm_focused', help='Mood del video')
    @click.option('--style', default='Dark Ambient', help='Estilo del audio')
    @click.option('--frame', default=None, type=click.Path(exists=True), help='Frame de video como fondo')
    @click.option('--output', default=None, help='Nombre de archivo')
    def main(hours, mood, style, frame, output):
        """Genera un thumbnail para MusicaYT."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        gen = ThumbnailGenerator(config)
        result = gen.generate(hours, mood, style, frame, output)
        print(f"✓ Thumbnail: {result}")

    main()

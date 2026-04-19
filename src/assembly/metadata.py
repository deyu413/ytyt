"""
MusicaYT — Generador de Metadata SEO
=====================================
Genera automáticamente títulos, descripciones y tags optimizados
para maximizar RPM y CTR en el nicho de trading.

Incluye:
- Fórmulas de títulos de alto rendimiento
- Descripciones con timestamps y enlaces de afiliados
- Tags de alto valor RPM (forex, crypto, trading)
"""

import os
import random
import logging
import math
from pathlib import Path
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """Genera metadata SEO optimizada para YouTube."""

    def __init__(self, config, prompts_data):
        self.config = config
        self.prompts_data = prompts_data
        self.meta_config = config.get('metadata', {})
        self.affiliates = config.get('affiliates', {})
        self.templates_dir = Path(config['paths']['templates'])

    def generate_metadata(self, audio_duration_hours, prompt_info, version=None,
                          title_family=None, description_mode='long'):
        """
        Genera metadata completa para un video.
        
        Args:
            audio_duration_hours: Duración del video en horas
            prompt_info: Info del prompt usado (category, mood, visual_style)
            version: Número de versión del video (para títulos tipo "v3.0")
            title_family: Familia de título forzada (session/emotional/identity/aesthetic)
                          Si None, se infiere del mood.
            description_mode: 'long' para vídeos importantes, 'short' para producción rápida
            
        Returns:
            dict con title, description, description_short, tags, category,
                  thumbnail_label, title_family
        """
        if version is None:
            version = f"{random.randint(1, 9)}.{random.randint(0, 9)}"

        duration_str = self._format_duration(audio_duration_hours)
        mood = prompt_info.get('mood', 'deep_focus')
        category = prompt_info.get('category', 'stoic_night')
        style = self._mood_to_style(mood)

        # Determinar familia de título
        if title_family is None:
            title_family = self._mood_to_title_family(mood)

        # 1. Generar título (sistema de 4 familias)
        title = self._generate_title(duration_str, mood, style, version, title_family)

        # 2. Generar descripción (larga y corta)
        description_long = self._generate_description_long(
            duration_str, audio_duration_hours, prompt_info, title
        )
        description_short = self._generate_description_short(
            duration_str, audio_duration_hours, prompt_info, title
        )

        # 3. Tags
        tags = self._generate_tags(mood, category)

        # 4. Label para thumbnail
        thumbnail_label = self._get_thumbnail_label(title_family)

        metadata = {
            'title': title,
            'description': description_long if description_mode == 'long' else description_short,
            'description_long': description_long,
            'description_short': description_short,
            'tags': tags,
            'category': self.meta_config.get('category', '22'),
            'language': self.meta_config.get('language', 'en'),
            'made_for_kids': False,
            'synthetic_content': True,
            'privacy_status': self.config.get('youtube', {}).get('privacy_status', 'public'),
            'title_family': title_family,
            'thumbnail_label': thumbnail_label,
        }

        logger.info(f"Metadata generada — Familia: {title_family}, Título: {title[:60]}...")
        return metadata

    def _generate_title(self, duration_str, mood, style, version, title_family='session'):
        """Genera un título usando el sistema de 4 familias."""
        families = self.prompts_data.get('title_families', {})

        # Obtener templates de la familia seleccionada
        templates = families.get(title_family, [])
        if not templates:
            # Fallback: usar cualquier familia disponible
            all_templates = []
            for fam in families.values():
                all_templates.extend(fam)
            templates = all_templates or ["{DURATION} Deep Focus for Live Trading"]

        template = random.choice(templates)

        title = template.replace("{DURATION}", duration_str)
        title = title.replace("{MOOD}", mood.replace("_", " ").title())
        title = title.replace("{STYLE}", style)
        title = title.replace("{VERSION}", f"v{version}")

        # Truncar a 100 chars (límite de YouTube)
        if len(title) > 100:
            title = title[:97] + "..."

        return title

    def _mood_to_title_family(self, mood):
        """Infiere la mejor familia de título según el mood del audio."""
        mapping = {
            # Moods nocturnos/estéticos → aesthetic
            'calm_focused': 'aesthetic',
            'deep_calm': 'aesthetic',
            # Moods de rendimiento → session
            'intense_focus': 'session',
            'operational': 'session',
            'pre_market': 'session',
            # Moods analíticos/pro → identity
            'analytical': 'identity',
            'peak_performance': 'identity',
            # Moods de control emocional → emotional
            'high_stakes': 'emotional',
            'abundance': 'emotional',
            'confident_calm': 'emotional',
        }
        return mapping.get(mood, 'session')

    def _get_thumbnail_label(self, title_family):
        """Obtiene un label corto (1-2 palabras) para el thumbnail."""
        labels = self.prompts_data.get('thumbnail_labels', {})
        family_labels = labels.get(title_family, ['DEEP FOCUS'])
        return random.choice(family_labels)

    # ═══════════════════════════════════════
    # DESCRIPCIÓN LARGA — para vídeos importantes
    # ═══════════════════════════════════════
    def _generate_description_long(self, duration_str, hours, prompt_info, title):
        """Descripción completa: timestamps, afiliados, credibilidad."""
        channel = self.meta_config.get('channel_name', 'TradeField Audio')
        mood_display = prompt_info.get('mood', 'deep_focus').replace('_', ' ').title()
        timestamps = self._generate_timestamps(hours)
        affiliate_lines = self._generate_affiliate_lines()

        description = f"""🎯 {title}

{channel} — Engineered Audio for Professional Traders

This {duration_str} ambient session is designed for maximum cognitive performance during live market sessions. 432Hz-retuned ambient design for calm focus and emotional control.

⚠️ This track is NOT entertainment — it's a productivity tool. Put it on, start your session, and let the sound do the work.

⏱️ TIMESTAMPS:
{timestamps}

═══════════════════════════════════════
📈 TRADING TOOLS I USE:
═══════════════════════════════════════
"""
        for line in affiliate_lines:
            description += f"{line}\n"

        description += f"""
═══════════════════════════════════════
🧠 OPTIMIZE YOUR SETUP:
═══════════════════════════════════════
🎧 Professional Focus Headphones — Link in pinned comment
💺 Ergonomic Trading Chair — Link in pinned comment
🖥️ Multi-Monitor Setup Guide — Link in pinned comment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️ About this track:
- Mood: {mood_display}
- Tuning: 432Hz-retuned ambient design for calm focus and emotional control
- Ad breaks: Every 60 minutes only (minimal disruption)
- AI-generated content (synthetic media)

🔔 Subscribe and enable notifications for daily focus sessions.

#trading #deepfocus #proptrading #432hz #daytrading #forex #ambient #concentration #flowstate #tradingmusic
"""
        return description.strip()

    # ═══════════════════════════════════════
    # DESCRIPCIÓN CORTA — para producción rápida
    # ═══════════════════════════════════════
    def _generate_description_short(self, duration_str, hours, prompt_info, title):
        """Descripción mínima para producción rápida de volumen."""
        channel = self.meta_config.get('channel_name', 'TradeField Audio')
        mood_display = prompt_info.get('mood', 'deep_focus').replace('_', ' ').title()
        affiliate_lines = self._generate_affiliate_lines()

        description = f"""🎯 {title}

{channel} — {duration_str} ambient session. 432Hz-retuned for calm focus and emotional control during live trading.

Put it on. Start your session. Let the sound do the work.

═══════════════════════════════════════
📈 TOOLS:
═══════════════════════════════════════
"""
        for line in affiliate_lines:
            description += f"{line}\n"

        description += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{mood_display} · 432Hz · Synthetic media

🔔 Subscribe for daily focus sessions.

#trading #deepfocus #432hz #daytrading #forex #ambient #tradingmusic
"""
        return description.strip()

    def _generate_timestamps(self, hours):
        """Genera timestamps automáticos (formato HH:MM:SS consistente)."""
        lines = []
        lines.append("00:00:00 — System Boot / Calibration Phase")
        lines.append("00:00:15 — Neural Entrainment (Beta Activation)")
        lines.append("00:05:00 — Steady State Deep Focus Begins")

        # Timestamps cada hora
        for hour in range(1, int(hours) + 1):
            lines.append(f"{hour:02d}:00:00 — Deep Focus Cycle {hour}")

        # Final
        end_h = int(hours)
        end_m = int((hours - end_h) * 60)
        lines.append(f"{end_h:02d}:{end_m:02d}:00 — Fade Out / Session End")

        return "\n".join(lines)

    def _generate_affiliate_lines(self):
        """Genera las líneas de afiliados desde la configuración."""
        lines = []

        trading = self.affiliates.get('trading', {})
        for key, info in trading.items():
            url = info.get('url', '')
            cta = info.get('cta', info.get('name', key))
            if url:
                lines.append(f"{cta}: {url}")
            else:
                lines.append(f"{cta}: [Link in pinned comment]")

        general = self.affiliates.get('general', {})
        for key, info in general.items():
            url = info.get('url', '')
            cta = info.get('cta', info.get('name', key))
            if url:
                lines.append(f"{cta}: {url}")

        return lines

    def _generate_tags(self, mood, category):
        """Genera tags optimizados para RPM alto."""
        base_tags = self.meta_config.get('default_tags', [])

        mood_tags = {
            'calm_focused': ['calm trading music', 'zen trading', 'emotional control trading'],
            'intense_focus': ['intense focus music', 'hyper focus', 'flow state music'],
            'analytical': ['analytical thinking music', 'logical reasoning ambient'],
            'peak_performance': ['peak performance audio', 'gamma waves', 'binaural beats focus'],
            'operational': ['trading floor sounds', 'office ambience', 'work ambient'],
            'high_stakes': ['high stakes trading', 'cyberpunk ambient', 'neon trading'],
            'pre_market': ['pre market routine', 'market open preparation'],
            'abundance': ['wealth meditation', 'abundance mindset', 'prosperity frequency'],
            'confident_calm': ['confidence building', 'morning trading routine'],
        }

        # High RPM keywords (fuerza anuncios caros)
        rpm_keywords = [
            'forex trading', 'cryptocurrency', 'stock market',
            'investment strategy', 'prop firm', 'funded trader',
            'algorithmic trading', 'trading psychology',
        ]

        all_tags = list(base_tags)
        all_tags.extend(mood_tags.get(mood, []))
        all_tags.extend(rpm_keywords)

        # Deduplicar y limitar a 500 chars total (límite YouTube)
        seen = set()
        unique_tags = []
        total_chars = 0
        for tag in all_tags:
            tag_lower = tag.lower()
            if tag_lower not in seen and total_chars + len(tag) < 480:
                seen.add(tag_lower)
                unique_tags.append(tag)
                total_chars += len(tag) + 1

        return unique_tags

    def _format_duration(self, hours):
        """Formatea la duración para títulos."""
        if hours == int(hours):
            return f"{int(hours)}-Hour"
        else:
            h = int(hours)
            m = int((hours - h) * 60)
            return f"{h}h{m:02d}m"

    def _mood_to_style(self, mood):
        """Convierte mood a estilo para títulos."""
        styles = {
            'calm_focused': 'Dark Ambient',
            'intense_focus': 'Deep Techno',
            'analytical': 'Binaural Alpha',
            'peak_performance': 'Gamma 40Hz',
            'operational': 'Lo-Fi Electronic',
            'high_stakes': 'Cyberpunk Ambient',
            'pre_market': 'Cinematic Dark',
            'abundance': 'Crystal 432Hz',
            'confident_calm': 'Warm Analog',
        }
        return styles.get(mood, 'Dark Ambient')


# --- CLI ---
if __name__ == "__main__":
    import click
    import json

    @click.command()
    @click.option('--hours', default=4, type=float, help='Duración del video en horas')
    @click.option('--category', default='stoic_night', help='Categoría de prompt')
    @click.option('--test', is_flag=True, help='Genera y muestra metadata de ejemplo')
    def main(hours, category, test):
        """Genera metadata SEO para MusicaYT."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts" / "trading_ambient.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)

        gen = MetadataGenerator(config, prompts_data)

        prompt_info = {
            'category': category,
            'mood': 'calm_focused',
            'visual_style': 'night_cityscape'
        }

        metadata = gen.generate_metadata(hours, prompt_info)

        if test:
            print("=" * 60)
            print("METADATA DE EJEMPLO")
            print("=" * 60)
            print(f"\n📌 TÍTULO ({len(metadata['title'])} chars):")
            print(f"   {metadata['title']}")
            print(f"\n📝 DESCRIPCIÓN ({len(metadata['description'])} chars):")
            print(metadata['description'])
            print(f"\n🏷️  TAGS ({len(metadata['tags'])}):")
            print(f"   {', '.join(metadata['tags'])}")
        else:
            print(json.dumps(metadata, indent=2, ensure_ascii=False))

    main()

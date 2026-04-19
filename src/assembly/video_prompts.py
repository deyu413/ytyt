"""
MusicaYT — Generador de Prompts para Video (Veo3)
===================================================
Genera prompts optimizados para que el usuario cree los clips
de fondo en Veo3 (Google). Los prompts están diseñados para:
- Movimiento sutil y lento (ideal para loops)
- Estética "Functional Luxury" alineada con el nicho de trading
- Ambientes que complementen el audio generado

El sistema genera prompts, el usuario los usa en Veo3,
y coloca los clips resultantes en assets/video_loops/
"""

import os
import random
import logging
from pathlib import Path
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# BIBLIOTECA DE PROMPTS PARA VEO3 — NICHO TRADING
# ═══════════════════════════════════════════════════════════
# Cada prompt está diseñado para generar video que:
# 1. Funcione bien en loop (movimiento continuo, sin inicio/final marcado)
# 2. Sea visualmente hipnótico sin ser distractivo
# 3. Comunique "lujo funcional" y "alta tecnología"
# ═══════════════════════════════════════════════════════════

VIDEO_PROMPT_LIBRARY = {
    "trading": {
        "night_cityscape": [
            {
                "name": "Tokyo Rain Trading Desk",
                "prompt": (
                    "Cinematic first-person view from inside a luxury penthouse office at night. "
                    "Floor-to-ceiling windows overlooking Tokyo's neon skyline. Heavy rain falling "
                    "on the glass, each drop catching city lights. Inside, a sleek dark hardwood "
                    "desk with two ultrawide monitors showing subtle candlestick charts (slightly "
                    "out of focus). A warm amber desk lamp casts soft light. Everything is still "
                    "except the rain and distant blinking city lights. Camera perfectly stationary. "
                    "Moody, atmospheric, cinematic color grading. 4K, shallow depth of field."
                ),
                "loop_tip": "El movimiento de la lluvia crea un loop natural perfecto.",
                "mood_match": ["calm_focused", "deep_calm"],
            },
            {
                "name": "NYC Skyline Night",
                "prompt": (
                    "Ultra slow cinematic aerial shot of Manhattan skyline at night. Camera "
                    "slowly drifting rightward, almost imperceptibly. Skyscrapers lit up with "
                    "warm office lights, streets below like rivers of light. Low clouds passing "
                    "between buildings. No people visible. Dark moody atmosphere with deep "
                    "blue and amber tones. Professional cinematography, anamorphic lens flares. "
                    "4K, film grain, 24fps."
                ),
                "loop_tip": "Movimiento lateral lento permite corte seamless.",
                "mood_match": ["high_stakes", "abundance"],
            },
            {
                "name": "London Financial District Dawn",
                "prompt": (
                    "Time-lapse style shot of London's Canary Wharf at pre-dawn. The sky "
                    "transitions from deep navy to golden hour. Office buildings light up floor "
                    "by floor. Reflections shimmer on the Thames. Fog rolls slowly across the "
                    "water. Very slow, hypnotic, and calming. No people, no text. Dark luxury "
                    "aesthetic. 4K cinematic quality."
                ),
                "loop_tip": "Ciclo amanecer crea transición natural para loop.",
                "mood_match": ["pre_market", "confident_calm"],
            },
        ],
        "luxury_trading_desk": [
            {
                "name": "Dark Trading Command Center",
                "prompt": (
                    "Close-up of a premium trading setup in a dark room. Three ultrawide curved "
                    "monitors displaying live trading charts with green and red candlesticks "
                    "(slightly blurred). Soft blue LED underglow. A matte black mechanical "
                    "keyboard. Subtle screen light reflections on a polished desk surface. "
                    "Occasional mouse movement (subtle). Everything is dark except the screens. "
                    "Cinematic shallow depth of field, teal and orange color grading. "
                    "First person POV, camera stationary. 4K."
                ),
                "loop_tip": "Los gráficos de trading en movimiento hacen loop natural.",
                "mood_match": ["operational", "intense_focus"],
            },
            {
                "name": "Minimalist Wealth Office",
                "prompt": (
                    "Wide shot of a minimalist luxury home office at night. Single large monitor "
                    "with trading platform (out of focus). Behind the desk, a full wall window "
                    "showing a rainy city skyline. Black Le Corbusier chair. A single orchid "
                    "on the desk. Subtle warm lighting from a hidden LED strip. The rain "
                    "outside provides all the movement. Ultra clean, no clutter. Dark mode "
                    "aesthetic. 4K, cinematic."
                ),
                "loop_tip": "La lluvia y las luces de la ciudad generan movimiento continuo.",
                "mood_match": ["calm_focused", "abundance"],
            },
        ],
        "data_visualization": [
            {
                "name": "Abstract Financial Data Flow",
                "prompt": (
                    "Abstract visualization of flowing financial data in 3D space. Glowing "
                    "green and cyan particles forming candlestick patterns that dissolve and "
                    "reform. Dark black background. Numbers and data points floating and fading. "
                    "Slow, hypnotic movement from left to right. No text legible, purely "
                    "abstract and aesthetic. Deep blue ambient glow. Looks like the inside of "
                    "a quantum computer analyzing markets. 4K, clean render."
                ),
                "loop_tip": "Partículas fluyendo en una dirección = loop perfecto.",
                "mood_match": ["analytical", "peak_performance"],
            },
            {
                "name": "Neural Network Trading",
                "prompt": (
                    "3D visualization of a neural network processing data. Interconnected "
                    "nodes pulsing with soft blue and white light. Data streams flowing between "
                    "nodes like synapses firing. Dark space background. Very slow, meditative "
                    "pace. Occasional brighter pulses travel across the network. Scientific "
                    "and sophisticated aesthetic. No text. 4K, clean."
                ),
                "loop_tip": "Las pulsaciones rítmicas de las neuronas permiten loop seamless.",
                "mood_match": ["analytical", "peak_performance"],
            },
        ],
        "nature_premium": [
            {
                "name": "Rain on Luxury Window",
                "prompt": (
                    "Extreme close-up of rain drops sliding down a floor-to-ceiling window at "
                    "night. Behind the glass, a blurry city skyline with warm amber and cool "
                    "blue lights. Each raindrop acts as a tiny lens, refracting the city lights. "
                    "Macro photography style, shallow depth of field. Very slow movement. "
                    "ASMR visual quality. No people, no text. 4K, cinematic."
                ),
                "loop_tip": "Las gotas de lluvia cayendo son el loop más natural que existe.",
                "mood_match": ["calm_focused", "deep_calm"],
            },
            {
                "name": "Ocean Waves Night",
                "prompt": (
                    "Slow motion aerial shot of dark ocean waves at night, lit by moonlight. "
                    "Waves rolling in from the horizon, white foam catching the light. Deep "
                    "dark blue water. No shoreline visible, just infinite ocean. Extremely "
                    "calming and hypnotic. Camera slowly descending toward the water surface. "
                    "Cinematic, moody, powerful but peaceful. 4K."
                ),
                "loop_tip": "Las olas son repetitivas por naturaleza — loop ideal.",
                "mood_match": ["abundance", "confident_calm"],
            },
        ],
    }
}


class VideoPromptGenerator:
    """Genera prompts optimizados para Veo3 basados en el mood del audio."""

    def __init__(self, config, prompts_data=None):
        self.config = config
        self.prompts_data = prompts_data
        self.niche = config['project']['niche']

    def get_prompt_for_mood(self, mood, category=None):
        """
        Selecciona el mejor prompt de video para un mood de audio específico.
        
        Args:
            mood: El mood del audio generado (ej: 'calm_focused')
            category: Categoría visual específica (opcional)
            
        Returns:
            dict con name, prompt, loop_tip, mood_match
        """
        niche_prompts = VIDEO_PROMPT_LIBRARY.get(self.niche, {})
        
        if not niche_prompts:
            logger.warning(f"No hay prompts de video para el nicho: {self.niche}")
            return None

        # Filtrar por categoría visual si se especifica
        if category and category in niche_prompts:
            candidates = niche_prompts[category]
        else:
            # Buscar en todas las categorías los que coincidan con el mood
            candidates = []
            for cat_name, cat_prompts in niche_prompts.items():
                for p in cat_prompts:
                    if mood in p.get('mood_match', []):
                        candidates.append({**p, 'visual_category': cat_name})

            # Si no hay coincidencia directa, usar todos
            if not candidates:
                for cat_name, cat_prompts in niche_prompts.items():
                    for p in cat_prompts:
                        candidates.append({**p, 'visual_category': cat_name})

        if not candidates:
            return None

        selected = random.choice(candidates)
        logger.info(f"Prompt de video seleccionado: {selected['name']}")
        return selected

    def generate_prompt_pair(self, mood):
        """
        Genera un par de prompts para Veo3 (el usuario necesita 2 clips).
        Selecciona 2 prompts complementarios que funcionen bien juntos en alternancia.
        
        Args:
            mood: El mood del audio
            
        Returns:
            Lista de 2 prompts de video
        """
        niche_prompts = VIDEO_PROMPT_LIBRARY.get(self.niche, {})
        all_candidates = []

        for cat_name, cat_prompts in niche_prompts.items():
            for p in cat_prompts:
                all_candidates.append({**p, 'visual_category': cat_name})

        # Priorizar los que coinciden con el mood
        matching = [c for c in all_candidates if mood in c.get('mood_match', [])]
        others = [c for c in all_candidates if mood not in c.get('mood_match', [])]

        pair = []

        # Primer prompt: coincidencia con el mood
        if matching:
            first = random.choice(matching)
            pair.append(first)
            matching.remove(first)
        elif others:
            first = random.choice(others)
            pair.append(first)
            others.remove(first)

        # Segundo prompt: diferente categoría visual para variedad
        remaining = matching + others
        if pair:
            different_cat = [c for c in remaining 
                           if c.get('visual_category') != pair[0].get('visual_category')]
            if different_cat:
                pair.append(random.choice(different_cat))
            elif remaining:
                pair.append(random.choice(remaining))

        # Si solo tenemos 1, duplicar con una nota
        if len(pair) == 1:
            pair.append(pair[0])
            logger.warning("Solo 1 prompt disponible, duplicando")

        return pair

    def format_for_user(self, prompts, audio_mood, audio_category):
        """
        Formatea los prompts de video para presentarlos al usuario.
        
        Returns:
            Texto formateado listo para copiar/pegar en Veo3
        """
        output = []
        output.append("=" * 60)
        output.append("🎬 PROMPTS DE VIDEO PARA VEO3")
        output.append(f"   Audio Mood: {audio_mood}")
        output.append(f"   Audio Category: {audio_category}")
        output.append("=" * 60)

        for i, p in enumerate(prompts, 1):
            output.append(f"\n{'─' * 50}")
            output.append(f"📹 CLIP {i}: {p['name']}")
            output.append(f"{'─' * 50}")
            output.append(f"\n{p['prompt']}")
            output.append(f"\n💡 Tip para loop: {p.get('loop_tip', 'N/A')}")
            output.append(f"🎨 Categoría visual: {p.get('visual_category', 'N/A')}")

        output.append(f"\n{'=' * 60}")
        output.append("📌 INSTRUCCIONES:")
        output.append("1. Copia cada prompt en Veo3")
        output.append("2. Genera clips de al menos 10 segundos")
        output.append("3. Descarga y coloca en: assets/video_loops/")
        output.append("4. Nombra los archivos: loop_1.mp4, loop_2.mp4")
        output.append("=" * 60)

        return "\n".join(output)


# --- CLI ---
if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--mood', default='calm_focused', help='Mood del audio')
    @click.option('--category', default=None, help='Categoría visual específica')
    @click.option('--save', default=None, help='Guardar prompts en archivo')
    def main(mood, category, save):
        """Genera prompts de video para Veo3."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        gen = VideoPromptGenerator(config)
        pair = gen.generate_prompt_pair(mood)
        
        formatted = gen.format_for_user(pair, mood, category or "auto")
        print(formatted)

        if save:
            save_path = Path(save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"\n✓ Guardado en: {save_path}")

    main()

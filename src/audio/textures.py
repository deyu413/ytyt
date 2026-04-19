"""
MusicaYT — Generador de Texturas Ambientales
=============================================
Genera y gestiona capas de textura (lluvia, servidores, ruido rosa)
que se superponen a la pista musical principal para añadir profundidad
y enmascarar micro-imperfecciones en las transiciones.

Motor: ACE-Step en modo DiT-only (<4GB VRAM)
Estrategia: generar una biblioteca reutilizable, no regenerar cada vez.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


class TextureGenerator:
    """Genera y gestiona una biblioteca de texturas ambientales reutilizables."""

    def __init__(self, config):
        self.config = config
        self.texture_dir = Path(config['paths']['textures'])
        self.texture_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.texture_dir / "catalog.json"
        self.catalog = self._load_catalog()

    def _load_catalog(self):
        """Carga el catálogo de texturas existentes."""
        if self.catalog_path.exists():
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"textures": {}}

    def _save_catalog(self):
        """Guarda el catálogo de texturas."""
        with open(self.catalog_path, 'w', encoding='utf-8') as f:
            json.dump(self.catalog, f, indent=2, ensure_ascii=False)

    def get_available_textures(self):
        """Retorna las texturas ya generadas y disponibles."""
        available = {}
        for name, info in self.catalog.get('textures', {}).items():
            path = Path(info['path'])
            if path.exists():
                available[name] = info
        return available

    def needs_generation(self, texture_name):
        """Verifica si una textura necesita ser generada."""
        # Si el usuario ha soltado un archivo manualmente, no necesitamos generar
        manual_path = self.texture_dir / f"{texture_name}.wav"
        if manual_path.exists():
            return False
            
        if texture_name not in self.catalog.get('textures', {}):
            return True
        path = Path(self.catalog['textures'][texture_name]['path'])
        return not path.exists()

    def generate_texture(self, texture_name, prompt, duration_min=30):
        """
        Genera una textura ambiental individual.
        
        Args:
            texture_name: Nombre identificador (ej: 'rain_on_glass')
            prompt: Prompt descriptivo para la generación
            duration_min: Duración en minutos
            
        Returns:
            Path al archivo generado
        """
        output_path = self.texture_dir / f"{texture_name}.wav"

        if output_path.exists() and texture_name in self.catalog.get('textures', {}):
            logger.info(f"Textura '{texture_name}' ya existe, reutilizando: {output_path}")
            return output_path

        logger.info(f"Generando textura: {texture_name} ({duration_min} min)")

        # Usar ACE-Step en modo DiT-only para máxima eficiencia
        from .generator import AudioGenerator
        gen = AudioGenerator(engine="ace_step")

        try:
            # ACE-Step entrenado idealmente para <= 60s. Limitamos texturas a 30s (bucle infinito después)
            actual_duration = 30
            result_path = gen.generate_segment(
                prompt_text=prompt,
                output_filename=f"texture_{texture_name}",
                duration_sec=actual_duration
            )

            # Mover al directorio de texturas si no está allí
            if result_path != output_path:
                import shutil
                shutil.move(str(result_path), str(output_path))

            # Actualizar catálogo
            self.catalog.setdefault('textures', {})[texture_name] = {
                'path': str(output_path),
                'prompt': prompt,
                'duration_min': duration_min,
                'generated_at': datetime.now().isoformat(),
            }
            self._save_catalog()

            logger.info(f"Textura generada y catalogada: {texture_name}")
            return output_path

        except Exception as e:
            logger.error(f"Error generando textura '{texture_name}': {e}")
            raise

    def generate_library(self, prompts_data):
        """
        Genera toda la biblioteca de texturas desde los prompts configurados.
        Solo genera las que faltan (incremental).
        
        Args:
            prompts_data: Datos de prompts cargados del YAML
            
        Returns:
            dict con nombre → path de todas las texturas disponibles
        """
        textures_config = prompts_data.get('textures', {})
        results = {}

        for name, tex_info in textures_config.items():
            if self.needs_generation(name):
                logger.info(f"Textura '{name}' no encontrada, generando...")
                try:
                    path = self.generate_texture(
                        texture_name=name,
                        prompt=tex_info['prompt'].strip(),
                        duration_min=tex_info.get('duration_min', 30)
                    )
                    results[name] = str(path)
                except Exception as e:
                    logger.error(f"No se pudo generar '{name}': {e}")
            else:
                # Si el usuario descargó manualmente el archivo pero no está en el catálogo, adoptarlo
                manual_path = self.texture_dir / f"{name}.wav"
                if name not in self.catalog.get('textures', {}) and manual_path.exists():
                    from datetime import datetime
                    self.catalog.setdefault('textures', {})[name] = {
                        'path': str(manual_path),
                        'prompt': "Textura agregada manualmente por el usuario",
                        'duration_min': tex_info.get('duration_min', 30),
                        'generated_at': datetime.now().isoformat()
                    }
                    self._save_catalog()
                    
                path = self.catalog['textures'][name]['path']
                results[name] = path
                logger.info(f"Textura '{name}' local detectada: {path}")

        logger.info(f"Biblioteca de texturas: {len(results)} disponibles")
        return results


# --- CLI ---
if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--texture', default=None, help='Generar una textura específica')
    @click.option('--all', 'gen_all', is_flag=True, help='Generar toda la biblioteca')
    @click.option('--list', 'list_textures', is_flag=True, help='Listar texturas disponibles')
    def main(texture, gen_all, list_textures):
        """Gestiona la biblioteca de texturas ambientales."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts" / "trading_ambient.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)

        tg = TextureGenerator(config)

        if list_textures:
            available = tg.get_available_textures()
            if available:
                print(f"\n📦 Texturas disponibles ({len(available)}):")
                for name, info in available.items():
                    print(f"  • {name} — {info.get('duration_min', '?')} min")
            else:
                print("\n⚠️  No hay texturas generadas todavía.")
            return

        if gen_all:
            results = tg.generate_library(prompts_data)
            print(f"\n✓ Biblioteca completa: {len(results)} texturas")
            return

        if texture:
            tex_data = prompts_data.get('textures', {}).get(texture)
            if not tex_data:
                print(f"❌ Textura '{texture}' no encontrada en prompts")
                return
            path = tg.generate_texture(texture, tex_data['prompt'], tex_data.get('duration_min', 30))
            print(f"✓ {path}")
            return

        print("Usa --help para ver opciones")

    main()

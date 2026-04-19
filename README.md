# 🎵 MusicaYT

**Pipeline automatizado de producción musical generativa para YouTube.**

Genera pistas ambient de **2-10 horas** optimizadas para el nicho de trading, con metadata SEO de alto RPM, thumbnails premium, y publicación automática en YouTube.

---

## 🏗️ Arquitectura

```
musciayt/
├── config/
│   ├── settings.yaml              # Config global
│   └── prompts/
│       └── trading_ambient.yaml   # Prompts por nicho
├── src/
│   ├── orchestrator.py            # 🎯 Pipeline maestro
│   ├── audio/
│   │   ├── generator.py           # Generación (ACE-Step / YuEGP)
│   │   ├── extender.py            # Extensión a horas (sliding window)
│   │   ├── textures.py            # Texturas ambientales (lluvia, etc.)
│   │   └── postprocess.py         # LUFS, 432Hz, limiter, mezcla
│   ├── assembly/
│   │   ├── composer.py            # Ensambla video (clips Veo3) + audio
│   │   ├── metadata.py            # SEO: títulos, tags, descripciones
│   │   ├── thumbnail.py           # Thumbnails 1280x720 dark mode
│   │   └── video_prompts.py       # Prompts para Veo3
│   └── publish/
│       └── uploader.py            # YouTube API v3 (OAuth2 + resumable)
├── sagemaker/
│   └── notebook_audio.py          # Script para SageMaker Studio Lab
├── assets/
│   ├── video_loops/               # 📹 Coloca aquí tus clips de Veo3
│   ├── audio_raw/                 # Segmentos generados
│   └── textures/                  # Texturas ambientales
├── output/                        # Videos finales, thumbnails, metadata
├── templates/
│   └── descriptions/trading.md    # Template de descripción
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

> También necesitas **FFmpeg** instalado: `winget install ffmpeg`

### 2. Generar música + metadata

```bash
python src/orchestrator.py --music-only --hours 4
```

Esto genera:
- ✅ Audio masterizado (4h, 432Hz, -14 LUFS)
- ✅ Thumbnail premium
- ✅ Metadata SEO (título, descripción, tags)
- ✅ Prompts de video para Veo3

### 3. Crear clips de video en Veo3

Usa los prompts generados (en `output/session_*_video_prompts.txt`) para crear 2 clips en [Veo3](https://deepmind.google/technologies/veo/). Coloca los clips en:

```
assets/video_loops/loop_1.mp4
assets/video_loops/loop_2.mp4
```

### 4. Ensamblar y subir a YouTube

```bash
# Dry run (valida sin subir)
python src/orchestrator.py --assemble-only --audio output/session_*_master.wav --dry-run

# Subir de verdad
python src/orchestrator.py --assemble-only --audio output/session_*_master.wav
```

---

## 📋 Modos del Orchestrator

| Modo | Comando | Qué hace |
|------|---------|----------|
| **Music-Only** | `--music-only` | Genera audio + metadata + prompts de video |
| **Full** | `--full` | Pipeline completo con upload a YouTube |
| **Assemble** | `--assemble-only --audio <path>` | Solo ensamblar y subir |

### Opciones adicionales

```bash
--hours 6          # Duración (2-10h, default: 4)
--category stoic   # Categoría de prompt específica
--segments 10      # Número de segmentos a generar
--dry-run          # Sin subir a YouTube
```

---

## 🎧 Pipeline de Audio

```
Prompts → ACE-Step/YuEGP → Segmentos (4min c/u)
                                    ↓
                    Sliding Window Extension (2-10h)
                                    ↓
                    Crossfade inteligente (zero-crossing)
                                    ↓
                    + Texturas (lluvia, server hum, pink noise)
                                    ↓
                    Post-procesamiento:
                      • Normalización LUFS (-14dB YouTube)
                      • Retuning 432Hz
                      • Soft limiter (tanh)
                      • Export WAV 24-bit + MP3 320kbps
```

---

## 💰 Monetización

- **Affiliate**: Apex Trader Funding, TradingView, NordVPN
- **AdSense**: Tags de alto RPM (forex, crypto, trading)
- **Mid-rolls**: Cada 60 min (mínima interrupción)

---

## ☁️ SageMaker (Audio Premium)

Para pistas de mayor calidad musical, usa YuEGP en SageMaker Studio Lab:

```bash
# En SageMaker (GPU gratuita T4 16GB, 4h/sesión)
python sagemaker/notebook_audio.py
```

---

## ⚙️ Configuración YouTube

1. Crea un proyecto en [Google Cloud Console](https://console.cloud.google.com/)
2. Habilita YouTube Data API v3
3. Crea credenciales OAuth2 (Desktop app)
4. Descarga `client_secret.json` → `config/client_secret.json`

---

## 📄 Licencia

Uso personal / interno.

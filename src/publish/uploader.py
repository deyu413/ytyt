"""
MusicaYT — YouTube Uploader
============================
Subida automatizada de videos a YouTube via API v3.
Configura metadata, thumbnail, marcado sintético y scheduling.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class YouTubeUploader:
    """Sube videos a YouTube con metadata optimizada."""

    def __init__(self, config):
        self.config = config
        self.yt_config = config.get('youtube', {})
        self.client_secrets = Path(self.yt_config.get('client_secrets_file', 'config/client_secret.json'))
        self.token_file = Path(self.yt_config.get('token_file', 'config/youtube_token.json'))
        self._service = None

    def _get_authenticated_service(self):
        """Autentica con YouTube API v3 usando OAuth2."""
        if self._service:
            return self._service

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            SCOPES = ['https://www.googleapis.com/auth/youtube.upload',
                       'https://www.googleapis.com/auth/youtube']

            creds = None

            if self.token_file.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                else:
                    if not self.client_secrets.exists():
                        raise FileNotFoundError(
                            f"Client secrets no encontrado: {self.client_secrets}\n"
                            f"Descarga las credenciales OAuth2 desde Google Cloud Console:\n"
                            f"https://console.cloud.google.com/apis/credentials"
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.client_secrets), SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Guardar token
                self.token_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.token_file, 'w') as f:
                    f.write(creds.to_json())

            self._service = build('youtube', 'v3', credentials=creds)
            logger.info("YouTube API autenticada correctamente")
            return self._service

        except ImportError:
            logger.error("Instala las dependencias: pip install google-api-python-client google-auth-oauthlib")
            raise

    def upload(self, video_path, metadata, thumbnail_path=None, schedule_time=None):
        """
        Sube un video a YouTube.
        
        Args:
            video_path: Path al video final
            metadata: dict con title, description, tags, category, etc.
            thumbnail_path: Path al thumbnail (opcional)
            schedule_time: datetime para publicación programada (opcional)
            
        Returns:
            dict con video_id y url del video subido
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        service = self._get_authenticated_service()

        # Preparar body
        privacy = metadata.get('privacy_status', 'public')
        if schedule_time:
            privacy = 'private'  # YouTube requiere private para scheduling

        body = {
            'snippet': {
                'title': metadata['title'],
                'description': metadata['description'],
                'tags': metadata.get('tags', []),
                'categoryId': metadata.get('category', '22'),
                'defaultLanguage': metadata.get('language', 'en'),
                'defaultAudioLanguage': metadata.get('language', 'en'),
            },
            'status': {
                'privacyStatus': privacy,
                'madeForKids': metadata.get('made_for_kids', False),
                'selfDeclaredMadeForKids': metadata.get('made_for_kids', False),
            }
        }

        # Scheduling
        if schedule_time:
            body['status']['publishAt'] = schedule_time.isoformat() + 'Z'
            body['status']['privacyStatus'] = 'private'

        # Marcado de contenido sintético (obligatorio 2026)
        if metadata.get('synthetic_content', True):
            body['status']['containsSyntheticMedia'] = True

        logger.info(f"Subiendo: {metadata['title'][:60]}...")
        logger.info(f"Archivo: {video_path} ({video_path.stat().st_size / (1024**3):.2f} GB)")

        try:
            from googleapiclient.http import MediaFileUpload

            media = MediaFileUpload(
                str(video_path),
                mimetype='video/mp4',
                resumable=True,
                chunksize=50 * 1024 * 1024  # 50MB chunks
            )

            request = service.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )

            # Upload con progreso
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"  Progreso: {progress}%")

            video_id = response['id']
            logger.info(f"Video subido: https://youtube.com/watch?v={video_id}")

            # Subir thumbnail
            if thumbnail_path and Path(thumbnail_path).exists():
                self._set_thumbnail(service, video_id, thumbnail_path)

            return {
                'video_id': video_id,
                'url': f"https://youtube.com/watch?v={video_id}",
                'title': metadata['title'],
                'status': privacy
            }

        except Exception as e:
            logger.error(f"Error en subida: {e}")
            raise

    def _set_thumbnail(self, service, video_id, thumbnail_path):
        """Sube el thumbnail personalizado."""
        try:
            from googleapiclient.http import MediaFileUpload

            media = MediaFileUpload(str(thumbnail_path), mimetype='image/png')
            service.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            logger.info(f"  Thumbnail configurado: {thumbnail_path}")

        except Exception as e:
            logger.warning(f"  Error subiendo thumbnail: {e}")

    def upload_dry_run(self, video_path, metadata, thumbnail_path=None):
        """
        Simulación de subida: valida metadata sin subir realmente.
        Útil para testing del pipeline.
        """
        video_path = Path(video_path)
        
        logger.info("=" * 40)
        logger.info("DRY RUN — No se subirá nada")
        logger.info("=" * 40)
        logger.info(f"Video: {video_path}")
        logger.info(f"  Existe: {video_path.exists()}")
        if video_path.exists():
            logger.info(f"  Tamaño: {video_path.stat().st_size / (1024**3):.2f} GB")
        logger.info(f"Título: {metadata.get('title', 'N/A')}")
        logger.info(f"  Longitud: {len(metadata.get('title', ''))} chars (max 100)")
        logger.info(f"Descripción: {len(metadata.get('description', ''))} chars")
        logger.info(f"Tags: {len(metadata.get('tags', []))} tags")
        logger.info(f"Thumbnail: {thumbnail_path}")
        logger.info(f"Sintético: {metadata.get('synthetic_content', True)}")
        logger.info(f"Privacidad: {metadata.get('privacy_status', 'public')}")

        # Validaciones
        errors = []
        if len(metadata.get('title', '')) > 100:
            errors.append("❌ Título excede 100 caracteres")
        if len(metadata.get('description', '')) > 5000:
            errors.append("❌ Descripción excede 5000 caracteres")
        total_tag_chars = sum(len(t) for t in metadata.get('tags', []))
        if total_tag_chars > 500:
            errors.append(f"❌ Tags exceden 500 caracteres ({total_tag_chars})")
        if not metadata.get('synthetic_content', True):
            errors.append("⚠️  Contenido no marcado como sintético (obligatorio 2026)")

        if errors:
            for e in errors:
                logger.warning(e)
        else:
            logger.info("✅ Metadata válida — listo para subir")

        return {'dry_run': True, 'valid': len(errors) == 0, 'errors': errors}


# --- CLI ---
if __name__ == "__main__":
    import click
    import yaml

    @click.command()
    @click.argument('video_path', type=click.Path(exists=True))
    @click.option('--metadata-json', type=click.Path(exists=True), help='JSON con metadata')
    @click.option('--thumbnail', type=click.Path(exists=True), help='Thumbnail')
    @click.option('--dry-run', is_flag=True, help='Simular sin subir')
    def main(video_path, metadata_json, thumbnail, dry_run):
        """Sube un video a YouTube."""
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        uploader = YouTubeUploader(config)

        if metadata_json:
            with open(metadata_json, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'title': 'Test Upload',
                'description': 'Test',
                'tags': ['test'],
                'synthetic_content': True
            }

        if dry_run:
            result = uploader.upload_dry_run(video_path, metadata, thumbnail)
        else:
            result = uploader.upload(video_path, metadata, thumbnail)

        print(json.dumps(result, indent=2))

    main()

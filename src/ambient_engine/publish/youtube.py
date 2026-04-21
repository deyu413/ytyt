from __future__ import annotations

import json
from pathlib import Path


class YouTubePublisher:
    def __init__(self, client_secrets: Path, token_file: Path) -> None:
        self.client_secrets = client_secrets
        self.token_file = token_file

    def dry_run(self, video_path: Path, metadata: dict[str, object], thumbnail_path: Path | None = None) -> dict[str, object]:
        issues = []
        if not video_path.exists():
            issues.append("Video file does not exist.")
        if len(str(metadata.get("title", ""))) > 100:
            issues.append("Title exceeds YouTube limit.")
        if len(str(metadata.get("description", ""))) > 5000:
            issues.append("Description exceeds YouTube limit.")
        return {
            "dry_run": True,
            "valid": len(issues) == 0,
            "issues": issues,
            "video_path": str(video_path),
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
        }

    def upload(self, video_path: Path, metadata: dict[str, object], thumbnail_path: Path | None = None) -> dict[str, object]:
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RuntimeError("Google API dependencies are not installed.") from exc

        scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube",
        ]
        creds = None
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), scopes)
        if not creds or not creds.valid:
            if not self.client_secrets.exists():
                raise FileNotFoundError(f"Client secrets not found: {self.client_secrets}")
            flow = InstalledAppFlow.from_client_secrets_file(str(self.client_secrets), scopes)
            creds = flow.run_local_server(port=0)
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(creds.to_json(), encoding="utf-8")

        service = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": metadata["title"],
                "description": metadata["description"],
                "tags": metadata.get("tags", []),
                "defaultLanguage": metadata.get("language", "en"),
            },
            "status": {
                "privacyStatus": metadata.get("privacy_status", "private"),
                "madeForKids": False,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": True,
            },
        }
        media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            _, response = request.next_chunk()
        result = {"video_id": response["id"], "url": f"https://youtube.com/watch?v={response['id']}"}
        if thumbnail_path and thumbnail_path.exists():
            service.thumbnails().set(
                videoId=response["id"],
                media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
            ).execute()
        return result


from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class ChunkStorage:
    """Object storage backend for uploaded audio chunks.

    Backed by Cloudflare R2 (S3-compatible) so that the FastAPI process
    accepting an upload and the background worker processing it don't need
    to share the same local disk. Falls back to "disabled" when R2 is not
    configured, in which case callers should keep using local temp files.
    """

    def __init__(
        self,
        *,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )

    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> None:
        fileobj.seek(0)
        try:
            self._client.upload_fileobj(fileobj, self.bucket_name, key)
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Не удалось загрузить файл в R2 (key={key}): {exc}") from exc

    def download_to_file(self, key: str, suffix: str = ".bin") -> Path:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.close()
        target = Path(temp.name)
        try:
            self._client.download_file(self.bucket_name, key, str(target))
        except (BotoCoreError, ClientError) as exc:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Не удалось скачать файл из R2 (key={key}): {exc}") from exc
        return target

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=key)
        except (BotoCoreError, ClientError) as exc:
            logger.warning(f"Не удалось удалить объект из R2 (key={key}): {exc}")


_storage_instance: Optional[ChunkStorage] = None
_storage_initialized = False


def get_chunk_storage() -> Optional[ChunkStorage]:
    """Returns a cached ChunkStorage, or None if R2 is not configured.

    Configuration is read lazily (not at import time) so tests and local
    dev without a .env don't need R2 credentials.
    """
    global _storage_instance, _storage_initialized
    if _storage_initialized:
        return _storage_instance

    _storage_initialized = True
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    endpoint_url = os.getenv("R2_ENDPOINT_URL")

    # .env.example ships with placeholder values (e.g. "your_cloudflare_account_id").
    # If someone copies it without filling these in, treat R2 as unconfigured
    # instead of trying to build a client with a bogus endpoint.
    placeholder_prefixes = ("your_", "REPLACE_ME")
    values = {
        "CLOUDFLARE_ACCOUNT_ID": account_id,
        "R2_ACCESS_KEY_ID": access_key_id,
        "R2_SECRET_ACCESS_KEY": secret_access_key,
        "R2_BUCKET_NAME": bucket_name,
    }
    placeholder_vars = [
        name for name, value in values.items()
        if value and value.startswith(placeholder_prefixes)
    ]
    if placeholder_vars:
        logger.warning(
            f"R2 storage vars look like unfilled placeholders ({', '.join(placeholder_vars)}); "
            "treating R2 as unconfigured and falling back to local temp files."
        )
        return None

    if not (account_id and access_key_id and secret_access_key and bucket_name):
        logger.info("R2 storage is not configured; falling back to local temp files for chunk uploads.")
        return None

    try:
        _storage_instance = ChunkStorage(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
            endpoint_url=endpoint_url,
        )
    except Exception as exc:
        logger.error(f"Failed to initialize R2 storage client, falling back to local temp files: {exc}")
        return None
    logger.info(f"R2 storage configured for bucket '{bucket_name}'.")
    return _storage_instance


def reset_chunk_storage_cache() -> None:
    """Test helper: forces get_chunk_storage() to re-read env vars."""
    global _storage_instance, _storage_initialized
    _storage_instance = None
    _storage_initialized = False

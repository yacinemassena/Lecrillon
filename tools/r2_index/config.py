from pathlib import Path

import boto3
from botocore.config import Config

R2_ACCOUNT_ID = "2a139e9393f803634546ad9d541d37b9"
R2_ACCESS_KEY_ID = "fdfa18bf64b18c61bbee64fda98ca20b"
R2_SECRET_ACCESS_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET_NAME = "europe"
R2_DATASETS_PREFIX = "datasets/"
DEFAULT_INDEX_DB_PATH = Path(__file__).resolve().parent / "r2_upload_index.sqlite3"


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto',
    )

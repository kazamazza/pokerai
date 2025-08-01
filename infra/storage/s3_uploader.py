import os

import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

class S3Uploader:
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        """
        Upload a file from local filesystem to S3 bucket.
        """
        try:
            self.s3.upload_file(str(local_path), self.bucket, s3_key)
            print(f"✅ Uploaded: {local_path} → s3://{self.bucket}/{s3_key}")
        except ClientError as e:
            print(f"❌ Upload failed: {e}")

    def download_file_if_missing(self, s3_key: str, local_path: Path) -> None:
        if local_path.exists():
            return
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, s3_key, str(local_path))
            print(f"✅ Downloaded: {s3_key} → {local_path}")
        except ClientError as e:
            print(f"❌ Failed to download {s3_key}: {e}")

    def download_file(self, s3_key: str, local_path: Path) -> None:
        """
        Download a file from S3 to local path.
        """
        try:
            self.s3.download_file(self.bucket, s3_key, str(local_path))
            print(f"✅ Downloaded: s3://{self.bucket}/{s3_key} → {local_path}")
        except ClientError as e:
            print(f"❌ Download failed: {e}")

    def list_files(self, prefix: str = "") -> list[str]:
        """
        List all files in the S3 bucket under a given prefix.
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = response.get("Contents", [])
            return [item["Key"] for item in contents]
        except ClientError as e:
            print(f"❌ List failed: {e}")
            return []
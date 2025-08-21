import gzip
import io
import json
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from utils.files import gunzip_file

load_dotenv()

class S3Client:
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

    def read_json_gz(self, s3_key: str) -> dict:
        """
        Read a .json.gz object from S3, decompress in-memory, and return parsed JSON.
        """
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            by = obj["Body"].read()
            with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
                text = gz.read().decode("utf-8")
            return json.loads(text)
        except ClientError as e:
            print(f"❌ Failed to read {s3_key}: {e}")
            raise
        except Exception as e:
            print(f"❌ Gzip/JSON parse failed for {s3_key}: {e}")
            raise

    def download_and_gunzip(self, s3_key: str, local_gz: Path, local_out: Path) -> Path:
        """
        Download s3_key → local_gz, then gunzip to local_out. Returns local_out.
        """
        self.download_file_if_missing(s3_key, local_gz)
        if not local_out.exists():
            gunzip_file(local_gz, local_out)
            print(f"✅ Unzipped: {local_gz} → {local_out}")
        return local_out
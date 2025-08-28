# tools/upload_monker_archive.py
import boto3
import os
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser(description="Upload monker.tar.gz archive to S3")
    ap.add_argument("--file", default="data/vendor/monker.tar.gz",
                    help="Path to local archive (default: data/vendor/monker.tar.gz)")
    ap.add_argument("--bucket", required=True, help="Target S3 bucket name")
    ap.add_argument("--prefix", default="data/vendor",
                    help="Target prefix inside bucket (default: data/vendor)")
    args = ap.parse_args()

    local_file = Path(args.file)
    if not local_file.exists():
        raise SystemExit(f"❌ File not found: {local_file}")

    s3_key = f"{args.prefix.rstrip('/')}/{local_file.name}"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    print(f"▶️ Uploading {local_file} → s3://{args.bucket}/{s3_key}")
    s3.upload_file(str(local_file), args.bucket, s3_key)
    print(f"✅ Upload complete: s3://{args.bucket}/{s3_key}")

if __name__ == "__main__":
    main()
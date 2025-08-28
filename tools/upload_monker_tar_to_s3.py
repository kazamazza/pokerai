#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError


def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )


def norm_key(prefix: str, member_name: str) -> str:
    # Make sure keys are clean and under the given prefix
    # Strip any leading './' and absolute bits from tar member names.
    name = member_name.lstrip("./")
    # If tar has a root folder, keep it (good for stacks like '15bb/...').
    # Build final S3 key
    return "/".join([prefix.rstrip("/"), name])


def should_upload(name: str, only_txt: bool) -> bool:
    if name.endswith("/"):
        return False  # directory
    if only_txt:
        return name.lower().endswith(".txt")
    return True


def main():
    import os
    ap = argparse.ArgumentParser(
        description="Upload expanded contents of a monker .tar.gz to S3 (without extracting to disk)."
    )
    ap.add_argument("--tar", type=str, default="data/vendor/monker.tar.gz",
                    help="Path to monker tar.gz")
    ap.add_argument("--bucket", type=str, default=os.getenv("AWS_BUCKET_NAME"),
                    help="S3 bucket (default from AWS_BUCKET_NAME)")
    ap.add_argument("--prefix", type=str, default="data/vendor/monker",
                    help="S3 key prefix where files will be uploaded")
    ap.add_argument("--only-txt", action="store_true", default=True,
                    help="Upload only .txt files (default: true)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded, but don’t upload")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only upload first N files")
    args = ap.parse_args()

    if not args.bucket:
        raise SystemExit("Set --bucket or AWS_BUCKET_NAME")

    tar_path = Path(args.tar)
    if not tar_path.exists():
        raise SystemExit(f"tarball not found: {tar_path}")

    s3 = s3_client()
    uploaded = 0
    skipped = 0
    total = 0

    # Stream each member directly to S3 (no local extraction)
    with tarfile.open(tar_path, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            total += 1
            if not should_upload(m.name, only_txt=args.only_txt):
                skipped += 1
                continue

            s3_key = norm_key(args.prefix, m.name)
            if args.dry_run:
                print(f"[dry-run] {m.name} → s3://{args.bucket}/{s3_key}")
            else:
                try:
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        print(f"⚠️  could not read member: {m.name}")
                        skipped += 1
                        continue
                    # upload file-like directly
                    s3.upload_fileobj(fobj, args.bucket, s3_key)
                    print(f"✅ {m.name} → s3://{args.bucket}/{s3_key}")
                except ClientError as e:
                    print(f"❌ upload failed for {m.name}: {e}")
                    skipped += 1
                    continue

            uploaded += 1
            if args.limit and uploaded >= args.limit:
                break

    print(f"\nDone. total_members={total} uploaded={uploaded} skipped={skipped} "
          f"dest=s3://{args.bucket}/{args.prefix.rstrip('/')}/")


if __name__ == "__main__":
    import os
    main()
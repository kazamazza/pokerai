import argparse
import os
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )


def is_sidecar(name: str) -> bool:
    """
    Detect macOS sidecars & junk:
      - __MACOSX/…
      - any path segment starting with '._'
      - .DS_Store
    """
    name = name.replace("\\", "/")
    if name.startswith("__MACOSX/"):
        return True
    parts = [p for p in name.split("/") if p]
    if any(p.startswith("._") for p in parts):
        return True
    if parts and parts[-1] == ".DS_Store":
        return True
    return False


def clean_member_name(member_name: str) -> str:
    """Normalize name: strip leading './', unify separators."""
    name = member_name.lstrip("./").replace("\\", "/")
    # drop empty segments
    parts = [p for p in name.split("/") if p]
    return "/".join(parts)


def norm_key(prefix: str, member_name: str, strip_root: bool) -> str:
    """
    Build final S3 key: <prefix>/<maybe-stripped member_name>
    If strip_root=True and member path has a top directory, remove it.
    """
    name = clean_member_name(member_name)
    parts = name.split("/")
    if strip_root and len(parts) > 1:
        parts = parts[1:]
    cleaned = "/".join(parts)
    return "/".join([prefix.rstrip("/"), cleaned])


def should_upload(name: str, only_txt: bool) -> bool:
    """Apply filters for sidecars & extension policy."""
    if is_sidecar(name):
        return False
    if only_txt and not name.lower().endswith(".txt"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Upload expanded contents of a Monker .tar.gz to S3 (streamed, no local extract)."
    )
    ap.add_argument("--tar", type=str, default="data/vendor/monker.tar.gz",
                    help="Path to monker tar.gz")
    ap.add_argument("--bucket", type=str, default=os.getenv("AWS_BUCKET_NAME"),
                    help="S3 bucket (default from AWS_BUCKET_NAME)")
    ap.add_argument("--prefix", type=str, default="data/vendor",
                    help="S3 key prefix (e.g. data/vendor)")
    ap.add_argument("--only-txt", action="store_true", default=True,
                    help="Upload only .txt files (default: true)")
    ap.add_argument("--strip-root-dir", action="store_true",
                    help="Strip the top-level folder inside the tar when forming keys")
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

    with tarfile.open(tar_path, mode="r:gz") as tf:
        # iterate members deterministically
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            total += 1
            raw_name = clean_member_name(m.name)

            if not should_upload(raw_name, only_txt=args.only_txt):
                skipped += 1
                continue

            s3_key = norm_key(args.prefix, raw_name, strip_root=args.strip_root_dir)

            if args.dry_run:
                print(f"[dry-run] {raw_name} → s3://{args.bucket}/{s3_key}")
            else:
                try:
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        print(f"⚠️  could not read member: {raw_name}")
                        skipped += 1
                        continue
                    s3.upload_fileobj(fobj, args.bucket, s3_key)
                    print(f"✅ {raw_name} → s3://{args.bucket}/{s3_key}")
                except ClientError as e:
                    print(f"❌ upload failed for {raw_name}: {e}")
                    skipped += 1
                    continue

            uploaded += 1
            if args.limit and uploaded >= args.limit:
                break

    print(f"\nDone. total_members={total} uploaded={uploaded} skipped={skipped} "
          f"dest=s3://{args.bucket}/{args.prefix.rstrip('/')}/")


if __name__ == "__main__":
    main()
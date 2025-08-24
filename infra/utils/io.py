import gzip
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from infra.storage.s3_client import S3Client


def compress_json_gzip(obj: dict) -> bytes:
    return gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"))

def parse_s3_url(url: str) -> tuple[str, str]:
    """
    s3://bucket/key  -> (bucket, key)
    """
    p = urlparse(url)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"Invalid S3 URL: {url}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    return bucket, key


def resolve_input_path(stake: int, inp: str | None, s3: S3Client | None = None) -> Path:
    """
    Returns a LOCAL **unzipped** path to decisions for the given stake.
    Accepts:
      - s3://bucket/key.jsonl.gz  (downloads & gunzips)
      - local .jsonl.gz           (gunzips)
      - local .jsonl              (as is)
      - None → defaults to decisions_nl{stake}.jsonl.gz in CWD
    """
    # choose defaults
    default_gz = Path(f"decisions_nl{stake}.jsonl.gz")
    default_jsonl = Path(f"decisions_nl{stake}.jsonl")

    if inp:
        if inp.startswith("s3://"):
            if s3 is None:
                raise ValueError("S3Client required to fetch s3:// input")
            bucket, key = parse_s3_url(inp)
            # if caller's S3Client has a fixed bucket, require it matches (optional)
            if s3.bucket and bucket and s3.bucket != bucket:
                raise ValueError(f"S3 bucket mismatch: resolver={bucket} client={s3.bucket}")

            local_gz = Path("tmp") / Path(key).name  # e.g., tmp/decisions_nl10.jsonl.gz
            local_jsonl = local_gz.with_suffix("")    # remove .gz → .jsonl
            s3.download_file_if_missing(key, local_gz)
            if not local_jsonl.exists():
                gunzip_file(local_gz, local_jsonl)
                print(f"✅ Unzipped: {local_gz} → {local_jsonl}")
            return local_jsonl

        # local path
        p = Path(inp)
        if not p.exists():
            raise FileNotFoundError(p)

        if p.suffix == ".gz":
            out = p.with_suffix("")  # strip .gz
            if not out.exists():
                gunzip_file(p, out)
                print(f"✅ Unzipped: {p} → {out}")
            return out

        # already uncompressed .jsonl
        return p

    # No --input provided: prefer local gz default
    if default_gz.exists():
        out = default_jsonl
        if not out.exists():
            gunzip_file(default_gz, out)
            print(f"✅ Unzipped: {default_gz} → {out}")
        return out

    # or fall back to plain .jsonl if present
    if default_jsonl.exists():
        return default_jsonl

    # last resort: try S3 default if caller passed an S3Client and uses std key layout
    if s3 is not None:
        key = f"parsed/{default_gz.name}"  # matches your uploader pattern
        local_gz = Path("tmp") / default_gz.name
        local_jsonl = Path("tmp") / default_jsonl.name
        s3.download_file_if_missing(key, local_gz)
        if not local_jsonl.exists():
            gunzip_file(local_gz, local_jsonl)
            print(f"✅ Unzipped: {local_gz} → {local_jsonl}")
        return local_jsonl

    raise FileNotFoundError(
        f"Could not resolve decisions file for stake NL{stake}. "
        f"Tried {default_gz}, {default_jsonl}, and S3 (if provided)."
    )
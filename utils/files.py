import gzip
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse


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

def gunzip_file(src_gz: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
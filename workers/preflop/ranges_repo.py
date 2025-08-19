# --- ranges_repo.py (or at top of your worker module) ---
import json, os, time, tempfile
from typing import Dict, Optional, Tuple

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# ENV config (override in prod)
VRM_BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
VRM_KEY    = os.getenv("VRM_KEY",    "villain_range_map.json")  # uploaded file name

_s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"}))

# In-memory cache
_VRM_MEM: Dict[str, str] = {}
_VRM_ETAG: Optional[str] = None
_VRM_LOADED_AT: float = 0.0

def _head_s3() -> Tuple[Optional[str], Optional[int]]:
    """Return (etag, content_length) for the map object or (None, None) if missing."""
    try:
        h = _s3.head_object(Bucket=VRM_BUCKET, Key=VRM_KEY)
        return h.get("ETag"), h.get("ContentLength")
    except ClientError:
        return None, None

def _download_s3_to_temp() -> str:
    """Download S3 object to a temp file and return the path."""
    fd, path = tempfile.mkstemp(prefix="vrm_", suffix=".json")
    os.close(fd)
    _s3.download_file(VRM_BUCKET, VRM_KEY, path)
    return path

def load_villain_range_map(force: bool = False) -> Dict[str, str]:
    """
    Ensure VRM is loaded in memory. Refresh if ETag changed (or force).
    Returns the in-memory dict.
    """
    global _VRM_MEM, _VRM_ETAG, _VRM_LOADED_AT

    etag, size = _head_s3()
    if etag is None or not size:
        raise FileNotFoundError(f"villain range map not found: s3://{VRM_BUCKET}/{VRM_KEY}")

    if not force and _VRM_MEM and _VRM_ETAG == etag:
        return _VRM_MEM  # already current

    # download & parse
    tmp = _download_s3_to_temp()
    try:
        with open(tmp, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            raise ValueError("range map json is empty or not a dict")
        _VRM_MEM = data
        _VRM_ETAG = etag
        _VRM_LOADED_AT = time.time()
        return _VRM_MEM
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

def get_vrm_entry(key: str) -> Optional[str]:
    """
    Look up a single entry. Ensures VRM is loaded once.
    """
    m = load_villain_range_map(force=False)
    return m.get(key)
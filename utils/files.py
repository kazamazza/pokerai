import gzip
import json

def compress_json_gzip(obj: dict) -> bytes:
    return gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
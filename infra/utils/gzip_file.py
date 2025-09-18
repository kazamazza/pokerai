import gzip
from pathlib import Path

def gzip_file(local_json: Path) -> Path:
    """
    Gzip a .json file → .json.gz next to it.
    Returns the Path to the .gz file.
    """
    gz_path = local_json.with_suffix(local_json.suffix + ".gz")
    with local_json.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        f_out.writelines(f_in)
    return gz_path
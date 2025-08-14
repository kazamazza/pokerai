import boto3, os
from pathlib import Path

REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
PREFIX = "preflop/ranges/"

s3 = boto3.client("s3", region_name=REGION)

def main():
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    print(f"S3 objects under {PREFIX}: {len(keys)}")
    Path("s3_keys.txt").write_text("\n".join(sorted(keys)) + "\n")
    print("Wrote s3_keys.txt")

if __name__ == "__main__":
    main()
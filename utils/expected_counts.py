import os
import json
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
EXPECTED_FILE_KEY = "expected_counts.json"

s3 = boto3.client("s3")

def load_expected_counts() -> dict:
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=EXPECTED_FILE_KEY)
        return json.loads(response["Body"].read())
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return {}
        raise

def update_expected_count(worker_key: str, count: int):
    data = load_expected_counts()
    data[worker_key] = count
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=EXPECTED_FILE_KEY,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json"
    )
    print(f"📌 Updated expected count: {worker_key} → {count}")
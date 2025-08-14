import os, boto3

REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
s3 = boto3.client("s3", region_name=REGION)

prefix = "preflop/ranges/profile=GTO/exploit=GTO/multiway=HU/pop=REGULAR/action=VS_OPEN/"
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)

print("----- VS_OPEN keys under prefix -----")
for obj in resp.get("Contents", []):
    print(obj["Key"])
# ml/etl/rangenet/postflop/submit_jobs.py
import sys
from pathlib import Path
import argparse, json, boto3, os
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/artifacts/rangenet_postflop_manifest.parquet")
    ap.add_argument("--queue_url", required=True)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    sqs = boto3.client("sqs",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    def to_msg(row):
        body = {
            "sha1": row["sha1"],
            "s3_key": row["s3_key"],
            # minimal payload workers need to reconstruct input.txt
            "params": {
                "street": int(row["street"]),
                "pot_bb": float(row["pot_bb"]),
                "effective_stack_bb": float(row["effective_stack_bb"]),
                "board": row["board"],
                "range_ip": row["range_ip"],
                "range_oop": row["range_oop"],
                "positions": row["positions"],
                "bet_sizing_id": row["bet_sizing_id"],
                "accuracy": float(row["accuracy"]),
                "max_iter": int(row["max_iter"]),
                "allin_threshold": float(row["allin_threshold"]),
                "node_key": row.get("node_key", "root"),
                "solver_version": row.get("solver_version", "v1"),
            },
        }
        return {"Id": row["sha1"][:80], "MessageBody": json.dumps(body)}

    # send in small batches
    batch, i = args.batch, 0
    while i < len(df):
        chunk = df.iloc[i:i+batch]
        entries = [to_msg(r) for _, r in chunk.iterrows()]
        sqs.send_message_batch(QueueUrl=args.queue_url, Entries=entries)
        i += batch
    print(f"✅ submitted {len(df)} jobs to SQS")

if __name__ == "__main__":
    main()
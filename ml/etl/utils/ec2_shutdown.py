import os
import time

import boto3
import requests


def _get_instance_id(timeout=1.5) -> str | None:
    try:
        r = requests.get("http://169.254.169.254/latest/meta-data/instance-id", timeout=timeout)
        return r.text.strip() if r.ok else None
    except Exception:
        return None

def _detect_region(timeout=1.5) -> str | None:
    try:
        r = requests.get("http://169.254.169.254/latest/dynamic/instance-identity/document", timeout=timeout)
        if r.ok:
            return r.json().get("region")
    except Exception:
        pass
    return None

def shutdown_ec2_instance(mode: str = "stop", wait_seconds: int = 5) -> None:
    iid = _get_instance_id()
    if not iid:
        print("ℹ️ Not on EC2 (or IMDS unreachable); skipping shutdown.")
        return
    reg = _detect_region() or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not reg:
        print("⚠️ Could not determine AWS region; skipping shutdown.")
        return

    print(f"🕘 Shutting down in {wait_seconds}s (mode={mode}) for {iid} in {reg}")
    time.sleep(wait_seconds)

    ec2 = boto3.client("ec2", region_name=reg)
    try:
        if mode == "terminate":
            ec2.terminate_instances(InstanceIds=[iid])
            print(f"🗑️ Terminate API sent for {iid}")
        else:
            ec2.stop_instances(InstanceIds=[iid])
            print(f"🛑 Stop API sent for {iid}")
    except Exception as e:
        print(f"❌ Failed to {mode} instance: {e}")
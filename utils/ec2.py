import subprocess
import urllib.request


def shutdown_instance():
    print("✅ All tasks completed. Shutting down EC2 instance...")
    subprocess.run(["sudo", "shutdown", "-h", "now"])

def is_ec2_instance() -> bool:
    try:
        with urllib.request.urlopen("http://169.254.169.254/latest/meta-data/instance-id", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False
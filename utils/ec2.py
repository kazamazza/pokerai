import subprocess


def shutdown_instance():
    print("✅ All tasks completed. Shutting down EC2 instance...")
    subprocess.run(["sudo", "shutdown", "-h", "now"])

def is_ec2_instance():
    try:
        with open("/sys/hypervisor/uuid") as f:
            return f.read().startswith("ec2")
    except FileNotFoundError:
        return False
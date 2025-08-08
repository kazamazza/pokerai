import os


def detect_threads(default_cpu_threads=1):
    # Respect CPU pinning from `taskset`
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1

    # Operator override beats heuristics
    override = os.getenv("WORKER_THREADS")
    if override:
        try:
            t = max(1, int(override))
            return t
        except ValueError:
            pass

    # Heuristic: for IO/mixed workloads, allow a small fan-out
    mode = (os.getenv("WORKER_MODE") or "").lower()  # "cpu" | "io" | "mixed"
    if mode in ("io", "mixed"):
        return min(4, max(2, cpus))

    # CPU-bound default: 1 thread per pinned CPU
    return default_cpu_threads
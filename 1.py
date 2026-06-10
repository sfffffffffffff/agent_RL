#!/usr/bin/env python3
# gpu_idle_filler.py

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)

    gpus = []
    for line in out.strip().splitlines():
        idx, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
        mem_used = int(mem_used)
        mem_total = int(mem_total)

        gpus.append(
            {
                "idx": int(idx),
                "util": int(util),
                "mem_used": mem_used,
                "mem_total": mem_total,
                "mem_free": mem_total - mem_used,
            }
        )

    return gpus


def launch_worker(gpu_idx, args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    script_path = str(Path(__file__).resolve())

    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--run-sec",
        str(args.max_run_sec),
        "--matrix-size",
        str(args.matrix_size),
        "--duty-cycle",
        str(args.duty_cycle),
    ]

    print(f"[sidecar] launch filler on GPU {gpu_idx}", flush=True)

    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdout=None,
        stderr=None,
    )


def kill_worker(proc):
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_scheduler(args):
    idle_since = {}
    workers = {}
    failure_cooldown = {}

    print("[sidecar] scheduler started", flush=True)
    print(f"[sidecar] low_util <= {args.low_util}%", flush=True)
    print(f"[sidecar] min_free_mb >= {args.min_free_mb}", flush=True)

    while True:
        now = time.time()

        try:
            gpus = query_gpus()
        except Exception as e:
            print(f"[sidecar] failed to query GPUs: {e}", flush=True)
            time.sleep(args.interval)
            continue

        for gpu in gpus:
            idx = gpu["idx"]

            if idx in workers:
                proc, start_time = workers[idx]

                if proc.poll() is not None:
                    runtime = now - start_time
                    print(f"[sidecar] worker on GPU {idx} finished after {runtime:.1f}s", flush=True)

                    if runtime < 5:
                        failure_cooldown[idx] = now

                    workers.pop(idx, None)
                    idle_since.pop(idx, None)
                    continue

                if now - start_time > args.max_run_sec + 10:
                    print(f"[sidecar] worker on GPU {idx} timeout, killing", flush=True)
                    kill_worker(proc)
                    workers.pop(idx, None)
                    idle_since.pop(idx, None)

                continue

            if idx in failure_cooldown and now - failure_cooldown[idx] < args.cooldown_sec:
                continue

            is_idle = (
                gpu["util"] <= args.low_util
                and gpu["mem_free"] >= args.min_free_mb
            )

            if is_idle:
                idle_since.setdefault(idx, now)

                if now - idle_since[idx] >= args.idle_sec:
                    proc = launch_worker(idx, args)
                    workers[idx] = (proc, now)
                    idle_since.pop(idx, None)
            else:
                idle_since.pop(idx, None)

        time.sleep(args.interval)


def run_worker(args):
    import torch

    if not torch.cuda.is_available():
        print("[worker] CUDA not available, exit", flush=True)
        return

    device = torch.device("cuda:0")

    matrix_size = args.matrix_size
    duty_cycle = max(0.01, min(args.duty_cycle, 1.0))
    run_sec = args.run_sec

    print(
        f"[worker] start on {torch.cuda.get_device_name(0)}, "
        f"matrix_size={matrix_size}, run_sec={run_sec}, duty_cycle={duty_cycle}",
        flush=True,
    )

    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)

    end_time = time.time() + run_sec
    period = 1.0

    while time.time() < end_time:
        active_until = time.time() + period * duty_cycle

        while time.time() < active_until:
            c = a @ b
            torch.cuda.synchronize()
            a, b = b, c

        sleep_time = period * (1.0 - duty_cycle)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("[worker] finished", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true")

    parser.add_argument("--low-util", type=int, default=20)
    parser.add_argument("--min-free-mb", type=int, default=10000)
    parser.add_argument("--idle-sec", type=int, default=30)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-run-sec", type=int, default=60)
    parser.add_argument("--cooldown-sec", type=int, default=120)

    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--duty-cycle", type=float, default=0.5)

    parser.add_argument("--run-sec", type=int, default=60)

    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_scheduler(args)


if __name__ == "__main__":
    main()

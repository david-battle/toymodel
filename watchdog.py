#!/usr/bin/env python3
"""Watchdog for training: monitors throughput, triggers checkpoint+restart on drop."""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
LOG_FILE = ROOT / "logs" / "pilot.log"
PID_FILE = ROOT / "logs" / "pilot.pid"
WATCHDOG_LOG = ROOT / "logs" / "watchdog.log"

DEFAULT_THRESHOLD = 10000
DEFAULT_CHECK_INTERVAL = 30
DEFAULT_RESTART_DELAY = 10

THROUGHPUT_RE = re.compile(r"\[step \d+\] tok=[\d.]+M loss=[\d.]+ lr=[\d.e-]+ scale=\d+ skip=\d+ tok/s=(\d+)")


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


def get_training_pid() -> int | None:
    """Get the training process PID from pid file or ps."""
    # First try pid file
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if pid_exists(pid):
                return pid
        except (ValueError, OSError):
            pass

    # Fallback: find via ps
    try:
        out = subprocess.check_output(
            ["ps", "aux"], text=True
        )
        for line in out.splitlines():
            if "train.py" in line and "grep" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    if pid_exists(pid):
                        return pid
    except subprocess.CalledProcessError:
        pass
    return None


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def parse_throughput(line: str) -> int | None:
    m = THROUGHPUT_RE.search(line)
    if m:
        return int(m.group(1))
    return None


def tail_log(filepath: Path):
    """Generator yielding new lines from a file as they're written using tail -f."""
    # Use tail -f which properly follows file appends from other processes
    proc = subprocess.Popen(
        ["tail", "-f", "-n", "0", str(filepath)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.terminate()
        proc.wait(timeout=2)


def send_sigterm(pid: int) -> bool:
    """Send SIGTERM and wait for process to exit. Returns True if exited cleanly."""
    log(f"Sending SIGTERM to training process {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        log(f"Failed to send SIGTERM: {e}")
        return False

    # Wait for process to exit (up to 60 seconds)
    for _ in range(60):
        if not pid_exists(pid):
            log(f"Training process {pid} exited")
            return True
        time.sleep(1)

    log(f"Training process {pid} did not exit after 60s, sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(2)
    return not pid_exists(pid)


def restart_training(resume_path: str, extra_args: list[str]) -> subprocess.Popen:
    """Restart training with --resume. Returns the new process."""
    cmd = [
        sys.executable, "train.py",
        "--resume", resume_path,
        *extra_args
    ]
    log(f"Restarting training: {' '.join(cmd)}")
    # Use the same pattern as run_training.sh: setsid + redirect + background
    # We need to run this in a shell to get the background PID
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    shell_cmd = f"setsid {cmd_str} >>{LOG_FILE} 2>&1 </dev/null & echo $!"
    proc = subprocess.Popen(
        shell_cmd,
        cwd=ROOT,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    stdout, stderr = proc.communicate(timeout=10)
    pid_output = stdout.decode().strip()
    if pid_output and pid_output.isdigit():
        pid = int(pid_output)
        PID_FILE.write_text(str(pid))
        log(f"Training restarted with PID {pid}")
        class DummyProc:
            def __init__(self, pid):
                self.pid = pid
        return DummyProc(pid)
    else:
        log(f"Failed to get training PID from shell: stdout={pid_output}, stderr={stderr.decode()}")
        return None


def parse_train_args() -> list[str]:
    """Extract the training arguments from the training process cmdline."""
    pid = get_training_pid()
    if pid:
        try:
            with open(f"/proc/{pid}/cmdline", "r") as f:
                cmdline = f.read().split("\x00")
            # Find train.py and extract args after it
            for i, part in enumerate(cmdline):
                if "train.py" in part:
                    return cmdline[i+1:]
        except (OSError, IndexError):
            pass
    # Fallback: full-run config
    return [
        "--n-layer", "12",
        "--n-head", "12",
        "--n-embd", "768",
        "--block", "512",
        "--budget", "2500000000",
        "--micro-batch", "4",
        "--accum", "64",
        "--eval-steps", "500",
        "--ckpt-steps", "500",
        "--log-steps", "20",
    ]


def main():
    ap = argparse.ArgumentParser(description="Training watchdog")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                    help="Throughput threshold (tok/s) to trigger restart")
    ap.add_argument("--interval", type=int, default=DEFAULT_CHECK_INTERVAL,
                    help="Check interval (seconds)")
    ap.add_argument("--restart-delay", type=int, default=DEFAULT_RESTART_DELAY,
                    help="Delay before restart after checkpoint (seconds)")
    ap.add_argument("--log-file", type=Path, default=LOG_FILE)
    ap.add_argument("--pid-file", type=Path, default=PID_FILE)
    args = ap.parse_args()

    log(f"Watchdog started: threshold={args.threshold} tok/s, "
        f"check_interval={args.interval}s, log={args.log_file}")

    train_args = parse_train_args()
    resume_path = str(ROOT / "ckpt" / "last.pt")

    # Initial PID
    pid = get_training_pid()
    if pid:
        log(f"Monitoring training process PID {pid}")
    else:
        log("No training process found at startup, starting training...")
        proc = restart_training(resume_path, train_args)
        pid = proc.pid

    last_throughput = None
    low_count = 0
    consecutive_low = 3  # Require N consecutive low readings

    for line in tail_log(args.log_file):
        # Check if training process still exists
        if pid and not pid_exists(pid):
            log(f"Training process {pid} died unexpectedly, restarting...")
            proc = restart_training(resume_path, train_args)
            pid = proc.pid
            low_count = 0
            continue
        elif not pid:
            log("No training process, restarting...")
            proc = restart_training(resume_path, train_args)
            pid = proc.pid
            low_count = 0
            continue

        # Parse throughput from log line
        tok_s = parse_throughput(line)
        if tok_s is not None:
            last_throughput = tok_s
            if tok_s < args.threshold:
                low_count += 1
                log(f"Low throughput detected: {tok_s} tok/s (count={low_count}/{consecutive_low})")
            else:
                low_count = 0

            # Trigger restart if threshold breached consistently
            if low_count >= consecutive_low:
                log(f"Throughput below {args.threshold} for {consecutive_low} consecutive checks. "
                    f"Triggering checkpoint+restart...")

                if pid:
                    if send_sigterm(pid):
                        log("Checkpoint saved, waiting before restart...")
                        time.sleep(args.restart_delay)

                        # Restart training
                        proc = restart_training(resume_path, train_args)
                        pid = proc.pid
                        low_count = 0
                    else:
                        log("Failed to gracefully stop training process")
                else:
                    log("No training PID to signal, attempting restart anyway...")
                    proc = restart_training(resume_path, train_args)
                    pid = proc.pid
                    low_count = 0

        # Periodic status log
        if last_throughput and int(time.time()) % 300 == 0:
            log(f"Status: pid={pid}, last_throughput={last_throughput} tok/s")


if __name__ == "__main__":
    main()
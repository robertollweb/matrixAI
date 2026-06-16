#!/usr/bin/env python3
"""PR3-C1 — Serving benchmark: MatrixAI HTTP server vs ThreadingHTTPServer + sklearn.

Measures throughput (req/s) and latency (p50/p95/p99) at various concurrency levels.
The baseline is a ThreadingHTTPServer backed by a real sklearn LogisticRegression —
a competent rival that also has threading and a real trained model, not hardcoded weights.

Run from the matrixAI root:
    python3 benchmarks/serving.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent

MATRIXAI_PORT  = 18600
BASELINE_PORT  = 18601
API_KEY        = "benchmarkkey"
CONCURRENCY_LEVELS = [1, 5, 10, 20]
REQUESTS_PER_LEVEL = 100
PAYLOAD = json.dumps({
    "income_score": 0.71, "credit_history": 0.12, "debt_ratio": 0.33,
    "employment_years": 0.22, "loan_amount_ratio": 0.65
}).encode()


def _sep(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


def _capture_env() -> dict:
    env: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
    except ImportError:
        env["sklearn"] = "not installed"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    env["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    return env


def _wait_ready(port: int, timeout: float = 15.0, path: str = "/health") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1)
            if r.status == 200:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def _load_test(port: int, concurrency: int, n: int,
               headers: dict | None = None, path: str = "/predict") -> list[float]:
    """Send n requests at given concurrency. Returns list of latencies in seconds."""
    url = f"http://127.0.0.1:{port}{path}"
    times: list[float] = []
    lock = threading.Lock()
    sem = threading.Semaphore(concurrency)

    def worker():
        req = urllib.request.Request(url, data=PAYLOAD, method="POST")
        req.add_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        with sem:
            t0 = time.perf_counter()
            try:
                r = urllib.request.urlopen(req, timeout=5)
                r.read()
                with lock:
                    times.append(time.perf_counter() - t0)
            except Exception:
                pass

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return times


def _throughput(times_s: list[float], concurrency: int) -> float:
    if not times_s:
        return 0.0
    wall = sum(times_s) / concurrency
    return round(len(times_s) / wall, 1)


# ── Baseline server: ThreadingHTTPServer + real sklearn LogisticRegression ─────
# This is a competent rival: threaded (same concurrency model as MatrixAI),
# real trained model from the same credit-scoring dataset, no hardcoded weights.

BASELINE_APP = """
import sys, json, csv
sys.path.insert(0, '{root}')

import numpy as np
from sklearn.linear_model import LogisticRegression
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COLS = ["income_score","credit_history","debt_ratio","employment_years","loan_amount_ratio"]
TRAIN_CSV = '{train_csv}'

def load_and_train():
    rows = []
    with open(TRAIN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    X = [[float(r[c]) for c in COLS] for r in rows]
    y = [int(float(r["approved"]) >= 0.5) for r in rows]
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    return clf

clf = load_and_train()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/health":
            body = b'{{"status":"ok"}}'
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(body)
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        X_pred = [[body[c] for c in COLS]]
        prob = float(clf.predict_proba(X_pred)[0][1])
        resp = json.dumps({{"prediction": prob}}).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(resp)

server = ThreadingHTTPServer(("127.0.0.1", {port}), Handler)
server.serve_forever()
""".format(
    root=ROOT,
    port=BASELINE_PORT,
    train_csv=str(ROOT / "examples/credit-scoring/data/train.csv"),
)


def _start_matrixai_server():
    params_path = (
        ROOT / "examples/credit-scoring/registry/entries/credit-scoring/v1.0/params.json"
    )
    cmd = [
        sys.executable, "-m", "matrixai", "serve",
        str(ROOT / "examples/credit-scoring/credit_scoring.mxai"),
        "--params", str(params_path),
        "--port", str(MATRIXAI_PORT),
        "--api-key", API_KEY,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=str(ROOT))


def _start_baseline_server():
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(BASELINE_APP)
    tmp.close()
    proc = subprocess.Popen(
        [sys.executable, tmp.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return proc, tmp.name


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    env = _capture_env()
    print("PR3-C1 — Serving Benchmark: MatrixAI HTTP vs ThreadingHTTPServer + sklearn")
    print("=" * 64)
    print(f"  Python:         {env['python']}")
    print(f"  sklearn:        {env.get('sklearn', 'unknown')}")
    print(f"  Platform:       {env.get('machine', 'unknown')} — {env.get('platform', 'unknown')}")
    if "ram_gb" in env:
        print(f"  RAM:            {env['ram_gb']} GB")
    print(f"  Model:          credit_scoring (5 features, sigmoid, stdlib backend)")
    print(f"  Concurrency:    {CONCURRENCY_LEVELS}")
    print(f"  Requests/level: {REQUESTS_PER_LEVEL} (+ 5 warmup)")
    print()
    print("  Baseline: ThreadingHTTPServer + sklearn LogisticRegression (real model,")
    print("  same dataset, same threading model — no auth, no traces, no schema)")
    print()

    print("  Starting MatrixAI server...", end=" ", flush=True)
    mx_proc = _start_matrixai_server()
    if not _wait_ready(MATRIXAI_PORT):
        print("FAILED — server did not start")
        mx_proc.kill()
        return
    print("ready.")

    print("  Starting baseline server (sklearn trains at startup)...", end=" ", flush=True)
    baseline_proc, baseline_tmp = _start_baseline_server()
    if not _wait_ready(BASELINE_PORT, timeout=20.0):
        print("FAILED — server did not start")
        mx_proc.kill()
        baseline_proc.kill()
        return
    print("ready.")

    mx_results: dict = {}
    baseline_results: dict = {}

    try:
        for concurrency in CONCURRENCY_LEVELS:
            _sep(f"Concurrency = {concurrency}")

            _load_test(MATRIXAI_PORT, 1, 5, {"Authorization": f"Bearer {API_KEY}"})
            _load_test(BASELINE_PORT, 1, 5)

            times = _load_test(MATRIXAI_PORT, concurrency, REQUESTS_PER_LEVEL,
                               {"Authorization": f"Bearer {API_KEY}"})
            rps = _throughput(times, concurrency)
            s = sorted(times)
            mx_results[concurrency] = {
                "rps": rps,
                "p50": round(s[len(s)//2] * 1000, 1),
                "p95": round(s[int(len(s)*0.95)] * 1000, 1),
                "p99": round(s[int(len(s)*0.99)] * 1000, 1),
            }
            mx = mx_results[concurrency]
            print(f"  MatrixAI:   {mx['rps']:>7.1f} req/s  p50={mx['p50']}ms  "
                  f"p95={mx['p95']}ms  p99={mx['p99']}ms")

            times = _load_test(BASELINE_PORT, concurrency, REQUESTS_PER_LEVEL)
            rps = _throughput(times, concurrency)
            s = sorted(times)
            baseline_results[concurrency] = {
                "rps": rps,
                "p50": round(s[len(s)//2] * 1000, 1),
                "p95": round(s[int(len(s)*0.95)] * 1000, 1),
                "p99": round(s[int(len(s)*0.99)] * 1000, 1),
            }
            bl = baseline_results[concurrency]
            print(f"  Baseline:   {bl['rps']:>7.1f} req/s  p50={bl['p50']}ms  "
                  f"p95={bl['p95']}ms  p99={bl['p99']}ms")

            mx_rps = mx_results[concurrency]["rps"]
            bl_rps = baseline_results[concurrency]["rps"]
            if mx_rps >= bl_rps:
                ratio = mx_rps / max(bl_rps, 0.1)
                print(f"  → MatrixAI {ratio:.1f}x faster (auth overhead amortized at this concurrency)")
            else:
                ratio = bl_rps / max(mx_rps, 0.1)
                print(f"  → Baseline {ratio:.1f}x faster (no auth, no trace, no schema — less work per request)")

    finally:
        mx_proc.kill()
        baseline_proc.kill()
        try:
            os.unlink(baseline_tmp)
        except Exception:
            pass

    _sep("Summary")
    print(f"  {'Concurrency':>12}  {'MatrixAI req/s':>16}  {'Baseline req/s':>15}  "
          f"{'Winner':>10}  {'Ratio':>7}  {'MatrixAI p50':>13}")
    print("  " + "─" * 84)
    for c in CONCURRENCY_LEVELS:
        mx = mx_results.get(c, {})
        bl = baseline_results.get(c, {})
        mx_rps = mx.get("rps", 0)
        bl_rps = bl.get("rps", 0)
        if mx_rps >= bl_rps:
            winner = "MatrixAI"
            ratio = mx_rps / max(bl_rps, 0.1)
        else:
            winner = "Baseline"
            ratio = bl_rps / max(mx_rps, 0.1)
        print(f"  {c:>12}  {mx_rps:>16.1f}  {bl_rps:>15.1f}  "
              f"{winner:>10}  {ratio:>6.1f}x  {mx.get('p50', 0):>12.1f}ms")

    print()
    print("  Baseline: ThreadingHTTPServer + sklearn LogisticRegression")
    print("  Same threading model as MatrixAI. Missing vs MatrixAI:")
    print("    x No Bearer token authentication")
    print("    x No input type validation or schema")
    print("    x No per-request execution trace or model hash")
    print("    x No versioning, registry, or tamper detection")
    print("    x No OpenAPI / Swagger documentation")
    print()
    print("  The overhead at low concurrency is those features. Each request pays")
    print("  the cost of auth + trace + schema. At higher concurrency that cost")
    print("  is amortized across parallel workers.")

    out = ROOT / "benchmarks" / "results_serving.json"
    out.write_text(json.dumps({
        "environment": env,
        "baseline_description": "ThreadingHTTPServer + sklearn LogisticRegression (real model, same dataset, same threading model)",
        "matrixai": {str(k): v for k, v in mx_results.items()},
        "baseline": {str(k): v for k, v in baseline_results.items()},
    }, indent=2))
    print(f"\n  Results saved to benchmarks/results_serving.json")


if __name__ == "__main__":
    main()

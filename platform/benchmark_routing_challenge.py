"""Benchmark the deployed routing classifier against saved Qwen v2 predictions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import threading
from time import perf_counter

import routing_model


ROOT = Path(__file__).resolve().parent
CHALLENGE = ROOT / "qlora_training/retraining_v2/data/challenge.jsonl"
QWEN_REPORT = ROOT / "qlora_training/llama_training/qwen_v2_reference/comparison.json"


def process_stats(pid):
    output = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "time=", "-o", "rss="], text=True,
    )
    cpu, rss = output.split()
    days, sep, remainder = cpu.partition("-")
    seconds = int(days) * 86400 if sep else 0
    seconds += sum(float(part) * 60 ** index
                   for index, part in enumerate(reversed((remainder if sep else days).split(":"))))
    return seconds, int(rss) / 1024


def percentile(values, p):
    return sorted(values)[math.ceil(p * len(values)) - 1]


def accuracy(cases):
    return sum(case["expected"] == case["predicted"] for case in cases) / len(cases) if cases else None


def measure(rows, repeats, manifest):
    worker = routing_model.embedding_worker
    stop = threading.Event()
    samples, errors = [], []
    sampler = None

    def predict(row):
        result = routing_model.predict(row["text"])
        if not result["available"] or result.get("model") != manifest["name"]:
            raise RuntimeError("The deployed MiniLM classifier was unavailable; refusing fallback timings.")
        return {"id": row["id"], "expected": row["domain"], "predicted": result["domain"],
                "accepted": result["accepted"], "auto_route": result["accepted"] and result["domain"] != "other"}

    def sample(pid):
        while not stop.wait(0.05):
            try:
                samples.append(process_stats(pid)[1])
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                errors.append(exc)
                return

    try:
        for row in rows[:10]:
            predict(row)
        pid = worker.process.pid
        cpu_start, rss_start = process_stats(pid)
        sampler = threading.Thread(target=sample, args=(pid,))
        sampler.start()
        latencies, cases = [], []
        start = perf_counter()
        for _ in range(repeats):
            current = []
            for row in rows:
                before = perf_counter()
                current.append(predict(row))
                latencies.append((perf_counter() - before) * 1000)
            if cases and cases != current:
                raise RuntimeError("Classifier predictions changed between timing passes.")
            cases = current
        elapsed = perf_counter() - start
        if worker.process.pid != pid:
            raise RuntimeError("Worker restarted during warm measurement.")
        cpu_end, rss_end = process_stats(pid)
        stop.set()
        sampler.join()
        if errors:
            raise RuntimeError("Unable to sample worker resource use.") from errors[0]
        return cases, {
            "warm_p50_ms": percentile(latencies, 0.50),
            "warm_p95_ms": percentile(latencies, 0.95),
            "timed_predictions": len(latencies),
            "worker_cpu_seconds": cpu_end - cpu_start,
            "mean_worker_cpu_percent_one_core_100": 100 * (cpu_end - cpu_start) / elapsed,
            "worker_peak_sampled_rss_mib": max(rss_start, rss_end, *samples),
            "rss_samples": len(samples) + 2,
            "timed_wall_seconds": elapsed,
        }
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Warm timing passes; accuracy uses unique rows.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; refuses to overwrite.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows = [json.loads(line) for line in CHALLENGE.read_text().splitlines() if line.strip()]
    reference = json.loads(QWEN_REPORT.read_text())
    qwen = reference["test"]["qlora"]
    qwen_cases = qwen["metrics"]["cases"]
    ids = [row["id"] for row in rows]
    if (len(ids) != 120 or len(set(ids)) != 120 or ids != qwen["engineering"]["ids"]
            or [(row["id"], row["domain"]) for row in rows]
            != [(case["id"], case["expected"]) for case in qwen_cases]):
        raise ValueError("Challenge IDs/order or labels differ from the saved Qwen v2 report.")
    if not math.isclose(accuracy(qwen_cases), qwen["metrics"]["accuracy"]):
        raise ValueError("Saved Qwen predictions do not match the reported accuracy.")
    manifest = routing_model.load_manifest()
    if not manifest or manifest["name"] != "frozen" or manifest["kind"] != "embedding":
        raise RuntimeError("Expected the deployed frozen embedding model, not a legacy fallback.")
    cases, timing = measure(rows, args.repeats, manifest)
    accepted = [case for case in cases if case["accepted"]]
    auto = [case for case in cases if case["auto_route"]]
    paired = {"both_correct": 0, "minilm_only_correct": 0, "qwen_only_correct": 0, "neither_correct": 0}
    for small, large in zip(cases, qwen_cases):
        first = small["expected"] == small["predicted"]
        second = large["expected"] == large["predicted"]
        paired["both_correct" if first and second else "minilm_only_correct" if first
               else "qwen_only_correct" if second else "neither_correct"] += 1
    engineering = qwen["engineering"]
    report = {
        "challenge": {"cases": len(rows), "sha256": hashlib.sha256(CHALLENGE.read_bytes()).hexdigest(),
                      "matching": "Exact saved IDs, order, and expected labels; synthetic, not human reviewed."},
        "host": {"system": platform.platform(), "architecture": platform.machine(),
                 "logical_cpus": os.cpu_count(), "torch_worker_threads": 4},
        "minilm": {**timing, "encoder": manifest["encoder_model"], "revision": manifest["encoder_revision"],
                   "head_sha256": manifest["head_sha256"], "accuracy": accuracy(cases),
                   "accepted_coverage": len(accepted) / len(cases), "accepted_accuracy": accuracy(accepted),
                   "auto_route_coverage": len(auto) / len(cases), "auto_route_accuracy": accuracy(auto)},
        "qwen_v2_saved": {key: engineering[key] for key in (
            "warm_p50_ms", "warm_p95_ms", "peak_process_rss_mib", "peak_gpu_allocated_mib", "latency_samples")},
        "paired_counts": paired,
        "scope": [
            "MiniLM: deployed predict() and CPU worker IPC, batch one, ten warmup requests.",
            "CPU/RSS cover only the MiniLM worker; CPU uses ps time deltas, RSS sampled about every 50 ms after warmup.",
            "Qwen: saved Colab T4 six-candidate scoring, no IPC, RSS includes load, CPU utilization not recorded.",
            "Accuracy uses top label, including other; policy, API, RAG and queue are excluded.",
        ],
        "cases": cases,
    }
    report["qwen_v2_saved"].update(accuracy=qwen["metrics"]["accuracy"], gpu=reference["hardware"]["gpu"])
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(output)
    summary = dict(report)
    del summary["cases"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

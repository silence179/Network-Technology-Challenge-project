from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from collections import deque


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "run_logs")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "project_experiment_manifest.json")
ALL_SUITES = (
    "reality_gap",
    "experiment1",
    "experiment2",
    "experiment3",
    "routing_benchmark",
    "otcp",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    command: list[str]
    cwd: str
    outputs: list[str]


def _relpath(path: str) -> str:
    return os.path.relpath(path, ROOT_DIR).replace("\\", "/")


def _build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    python = sys.executable
    specs: list[ExperimentSpec] = []

    if "reality_gap" in args.only:
        specs.append(
            ExperimentSpec(
                name="reality_gap",
                description="Experiment 0: reality gap validation",
                command=[python, os.path.join("experiment0_results", "experiment0_reality_gap.py")],
                cwd=ROOT_DIR,
                outputs=[os.path.join(ROOT_DIR, "experiment0_results")],
            )
        )

    if "experiment1" in args.only:
        specs.append(
            ExperimentSpec(
                name="experiment1",
                description="Experiment 1: cache-routing synergy",
                command=[python, os.path.join("experiment1_results", "experiment1_cache_routing.py")],
                cwd=ROOT_DIR,
                outputs=[os.path.join(ROOT_DIR, "experiment1_results")],
            )
        )

    if "experiment2" in args.only:
        specs.append(
            ExperimentSpec(
                name="experiment2",
                description="Experiment 2: topology stability",
                command=[python, os.path.join("experiment2_results", "experiment2_topology_stability.py")],
                cwd=ROOT_DIR,
                outputs=[os.path.join(ROOT_DIR, "experiment2_results")],
            )
        )

    if "experiment3" in args.only:
        specs.append(
            ExperimentSpec(
                name="experiment3",
                description="Experiment 3: UAV relay",
                command=[python, os.path.join("experiment3_results", "experiment3_uav_relay.py")],
                cwd=ROOT_DIR,
                outputs=[os.path.join(ROOT_DIR, "experiment3_results")],
            )
        )

    if "routing_benchmark" in args.only:
        specs.append(
            ExperimentSpec(
                name="routing_benchmark",
                description="Routing-family benchmark on s3_routing_core",
                command=[
                    python,
                    os.path.join("algorithms", "benchmark_algorithms.py"),
                    "traces/sat_trace",
                    "--max-steps",
                    str(args.routing_max_steps),
                    "--save-outputs",
                ],
                cwd=ROOT_DIR,
                outputs=[os.path.join(ROOT_DIR, "outputs", "benchmark_results")],
            )
        )

    if "otcp" in args.only:
        otcp_command = [
            python,
            "-m",
            "code.experiment",
            "--mode",
            args.otcp_mode,
            "--sat-dir",
            os.path.join("..", "traces", "sat_trace_100"),
            "--max-steps",
            str(args.otcp_max_steps),
        ]
        if args.otcp_output_suffix:
            otcp_command.extend(["--output-suffix", args.otcp_output_suffix])
        specs.append(
            ExperimentSpec(
                name="otcp",
                description="Cache-placement baseline suite from algorithms/code",
                command=otcp_command,
                cwd=os.path.join(ROOT_DIR, "algorithms"),
                outputs=[
                    os.path.join(ROOT_DIR, "figures"),
                    os.path.join(ROOT_DIR, "results"),
                ],
            )
        )

    return specs


def _run_experiment(spec: ExperimentSpec) -> dict:
    start_time = time.perf_counter()
    log_path = os.path.join(LOG_DIR, f"{spec.name}.log")
    log_tail = deque(maxlen=20)

    with open(log_path, "w", encoding="utf-8") as log_handle:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            spec.command,
            cwd=spec.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )

        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                log_handle.write(line)
                log_tail.append(line.rstrip())

        returncode = process.wait()

    runtime_s = round(time.perf_counter() - start_time, 3)
    existing_outputs = [_relpath(path) for path in spec.outputs if os.path.exists(path)]
    return {
        "name": spec.name,
        "description": spec.description,
        "command": " ".join(spec.command),
        "cwd": _relpath(spec.cwd),
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "runtime_s": runtime_s,
        "outputs": existing_outputs,
        "log_path": _relpath(log_path),
        "log_tail": list(log_tail),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full S3 experiment matrix.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=ALL_SUITES,
        default=list(ALL_SUITES),
        help="Run only a subset of suites.",
    )
    parser.add_argument("--routing-max-steps", type=int, default=200)
    parser.add_argument("--otcp-mode", choices=["main", "ablation", "scale", "zipf", "capacity", "all"], default="all")
    parser.add_argument("--otcp-max-steps", type=int, default=100)
    parser.add_argument("--otcp-output-suffix", default="")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    specs = _build_specs(args)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suites": [],
    }

    for spec in specs:
        print(f"\n=== Running {spec.name}: {spec.description} ===")
        result = _run_experiment(spec)
        manifest["suites"].append(result)
        print(f"Status: {result['status']}  Runtime: {result['runtime_s']:.3f}s")
        if result["outputs"]:
            print("Outputs:")
            for output in result["outputs"]:
                print(f"  - {output}")
        if result["status"] != "success" and not args.continue_on_error:
            break

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    failed = [suite for suite in manifest["suites"] if suite["status"] != "success"]
    print(f"\nManifest: {_relpath(MANIFEST_PATH)}")
    if failed:
        print("Failed suites:")
        for suite in failed:
            print(f"  - {suite['name']}")
        return 1
    print("All requested suites completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
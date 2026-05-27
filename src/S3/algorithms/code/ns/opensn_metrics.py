"""Collect OpenSN emulation metrics for the OTCP paper.

The script talks to an already running local OpenSN NodeDaemon, replays the
6x11 test topology, waits until all instances and links are active, and writes
the measured control-plane/data-plane metrics to JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_EMU_CONFIG = Path(
    "/home/xcroi/Desktop/OTCP/OpenSN-Library/example/test_topologies/emu_config_local.json"
)
DEFAULT_TOPOLOGY = Path(
    "/home/xcroi/Desktop/OTCP/OpenSN-Library/example/test_topologies/topology_config_6_11.json"
)
DEFAULT_OUTPUT = Path(
    "/home/xcroi/Desktop/OTCP/paper/results/opensn_metrics.json"
)
DEFAULT_HTTP_TIMEOUT = 60.0
DEFAULT_HTTP_RETRIES = 3


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: str | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    retries: int = DEFAULT_HTTP_RETRIES,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        req = urllib.request.Request(base_url + path, method=method)
        data = None
        if payload is not None:
            req.add_header("Content-Type", "application/json")
            data = payload.encode()
        try:
            with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
                return json.load(resp)
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < max(retries, 1):
                time.sleep(retry_delay)
    if last_error is None:
        raise RuntimeError(f"Failed to read OpenSN response for {path}")
    raise last_error


def timed_request(base_url: str, path: str, *, method: str = "GET", payload: str | None = None) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    data = request_json(base_url, path, method=method, payload=payload)
    return data, time.perf_counter() - start


def docker_status_counts(expected_names: set[str] | None = None) -> Counter:
    output = subprocess.check_output(
        ["docker", "ps", "-a", "--format", "{{.Names}} {{.Status}}"],
        text=True,
    )
    counts: Counter[str] = Counter()
    for line in output.splitlines():
        if not line.startswith(("Satellite_", "GroundStation_", "GroundTerminal_")):
            continue
        name, _, status = line.partition(" ")
        if expected_names is not None and name not in expected_names:
            continue
        if status.startswith("Up "):
            counts["Up"] += 1
        elif status.startswith("Created"):
            counts["Created"] += 1
        else:
            counts[status] += 1
    return counts


def start_created_containers() -> int:
    output = subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", "status=created"],
        text=True,
    ).strip()
    if not output:
        return 0

    container_ids = [line.strip() for line in output.splitlines() if line.strip()]
    if not container_ids:
        return 0

    subprocess.run(["docker", "start", *container_ids], check=False, capture_output=True, text=True)
    return len(container_ids)


def wait_for_platform(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            request_json(base_url, "/api/platform/status")
            return
        except Exception as exc:  # pragma: no cover - transient startup path
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"OpenSN platform is not reachable: {last_error}")


def wait_for_activation(base_url: str, expected_instances: int, expected_links: int, timeout_s: float) -> tuple[dict[str, Any], float]:
    deadline = time.time() + timeout_s
    start = time.perf_counter()
    last_snapshot: dict[str, Any] = {}

    while time.time() < deadline:
        try:
            node_data = request_json(base_url, "/api/node/")
            instance_data = request_json(base_url, "/api/instance/")
            link_data = request_json(base_url, "/api/link/")
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_snapshot = {"request_error": str(exc)}
            time.sleep(1.0)
            continue

        nodes = node_data.get("data") or []
        instances = instance_data.get("data") or []
        links = link_data.get("data") or []
        started_instances = sum(1 for item in instances if item.get("start"))
        enabled_links = sum(1 for item in links if item.get("enable"))
        expected_names = {item.get("name") for item in instances if item.get("name")}
        docker_counts = docker_status_counts(expected_names)

        last_snapshot = {
            "node_count": len(nodes),
            "instance_count": len(instances),
            "link_count": len(links),
            "started_instances": started_instances,
            "enabled_links": enabled_links,
            "docker_status": dict(docker_counts),
            "node_free_instance": nodes[0].get("free_instance") if nodes else None,
            "instance_type_counts": dict(Counter(item.get("type", "unknown") for item in instances)),
        }

        docker_total = sum(docker_counts.values())
        docker_ready = docker_total == expected_instances and docker_counts.get("Up", 0) == expected_instances

        if (
            not docker_ready
            and docker_counts.get("Created", 0) > 0
            and len(instances) == expected_instances
            and len(links) == expected_links
            and started_instances == expected_instances
            and enabled_links == expected_links
        ):
            started_created = start_created_containers()
            if started_created > 0:
                time.sleep(0.5)
                continue

        if (
            len(instances) == expected_instances
            and len(links) == expected_links
            and started_instances == expected_instances
            and enabled_links == expected_links
            and docker_ready
        ):
            return last_snapshot, time.perf_counter() - start

        time.sleep(0.2)

    raise RuntimeError(f"Timed out waiting for OpenSN activation: {last_snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OpenSN emulation metrics.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--emu-config", type=Path, default=DEFAULT_EMU_CONFIG)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--expected-instances", type=int, default=68)
    parser.add_argument("--expected-links", type=int, default=121)
    args = parser.parse_args()

    wait_for_platform(args.base_url, args.timeout)

    metrics: dict[str, Any] = {
        "platform": "OpenSN",
        "topology": "6x11",
        "emulation_config": str(args.emu_config),
        "topology_file": str(args.topology),
    }

    _, metrics["reset_seconds"] = timed_request(args.base_url, "/api/emulation/reset", method="POST", payload="")
    _, metrics["update_seconds"] = timed_request(
        args.base_url,
        "/api/emulation/update",
        method="POST",
        payload=args.emu_config.read_text(),
    )
    _, metrics["topology_seconds"] = timed_request(
        args.base_url,
        "/api/emulation/topology",
        method="POST",
        payload=args.topology.read_text(),
    )
    _, metrics["start_seconds"] = timed_request(args.base_url, "/api/emulation/start", method="POST", payload="")

    activation_snapshot, metrics["activation_seconds"] = wait_for_activation(
        args.base_url,
        args.expected_instances,
        args.expected_links,
        args.timeout,
    )

    metrics.update(activation_snapshot)
    metrics["total_replay_seconds"] = (
        metrics["reset_seconds"]
        + metrics["update_seconds"]
        + metrics["topology_seconds"]
        + metrics["start_seconds"]
        + metrics["activation_seconds"]
    )

    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
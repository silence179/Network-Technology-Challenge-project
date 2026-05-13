"""Run a multi-method OpenSN benchmark with realistic workload pressure.

Usage:
    python -m code.ns.opensn_benchmark
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

from .opensn_otcp_integration import (
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUNTIME_STATUS_DIR,
    SUPPORTED_METHODS,
    _count_request_failures,
    _normalize_method,
    run_workflow,
)


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / 'opensn_benchmark_realistic'


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _runtime_payload_is_complete(payload: dict[str, Any], expected_requests: int) -> bool:
    if not payload or payload.get('status'):
        return False

    if payload.get('runtime_complete') is False:
        return False

    status_file_count = int(payload.get('status_file_count', 0))
    live_instance_count = int(payload.get('live_instance_count', 0))
    pending_requests = int(payload.get('pending_requests', 0))
    evaluated_requests = int(payload.get('evaluated_requests', sum(payload.get('service_results', {}).values())))
    return (
        status_file_count > 0
        and live_instance_count > 0
        and status_file_count >= live_instance_count
        and pending_requests >= expected_requests
        and evaluated_requests >= expected_requests
    )


def _matches_expected_workload(manifest: dict[str, Any], expected_workload: dict[str, Any] | None) -> bool:
    if expected_workload is None:
        return True

    comparable_fields = {
        'method': manifest.get('method'),
        'requests_per_step': manifest.get('requests_per_step'),
        'cache_capacity': manifest.get('cache_capacity'),
        'zipf_alpha': manifest.get('zipf_alpha'),
        'migration_budget': manifest.get('migration_budget'),
        'max_satellites': manifest.get('max_satellites'),
        'max_uavs': manifest.get('max_uavs'),
        'sat_dir': manifest.get('sat_dir'),
        'seed': manifest.get('request_seed_base'),
    }
    for key, expected in expected_workload.items():
        actual = comparable_fields.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-9:
                return False
            continue
        if actual != expected:
            return False
    return True


def _expected_workload(
    *,
    method: str,
    requests_per_step: int,
    cache_capacity: int,
    zipf_alpha: float,
    migration_budget: int,
    max_satellites: int,
    max_uavs: int,
    sat_dir: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        'method': method,
        'requests_per_step': requests_per_step,
        'cache_capacity': cache_capacity,
        'zipf_alpha': zipf_alpha,
        'migration_budget': migration_budget,
        'max_satellites': max_satellites,
        'max_uavs': max_uavs,
        'sat_dir': str(sat_dir),
        'seed': seed,
    }


def _is_complete_method(method_dir: Path, apply_steps: int, expected_workload: dict[str, Any] | None = None) -> bool:
    replay_path = method_dir / 'replay_metrics.json'
    manifest_path = method_dir / 'sequence_manifest.json'
    if not replay_path.exists() or not manifest_path.exists():
        return False

    manifest = _load_json(manifest_path)
    if not _matches_expected_workload(manifest, expected_workload):
        return False
    replay = _load_json(replay_path)
    replay_steps = replay.get('steps', [])
    required_steps = max(apply_steps, 1)
    if len(replay_steps) < required_steps:
        return False

    manifest_steps = manifest.get('steps', [])[:required_steps]
    if len(manifest_steps) < required_steps:
        return False

    for entry in manifest_steps:
        step = int(entry['step'])
        runtime_path = method_dir / f'runtime_step_{step:03d}.json'
        if not runtime_path.exists():
            return False
        runtime_payload = _load_json(runtime_path)
        if not _runtime_payload_is_complete(runtime_payload, int(entry.get('request_count', 0))):
            return False

    return True


def _aggregate_method(method: str, method_dir: Path) -> dict[str, Any]:
    manifest = _load_json(method_dir / 'sequence_manifest.json')
    replay = _load_json(method_dir / 'replay_metrics.json')

    runtime_payloads = []
    incomplete_steps = []
    for entry in manifest.get('steps', []):
        step = int(entry['step'])
        runtime_path = method_dir / f'runtime_step_{step:03d}.json'
        if not runtime_path.exists():
            incomplete_steps.append(step)
            continue

        payload = _load_json(runtime_path)
        if not _runtime_payload_is_complete(payload, int(entry.get('request_count', 0))):
            incomplete_steps.append(step)
            continue
        runtime_payloads.append(payload)

    totals = Counter()
    service_results = Counter()
    per_step_hit_rates = []

    for payload in runtime_payloads:
        totals['local_hits'] += int(payload.get('local_hits', 0))
        totals['adjacent_hits'] += int(payload.get('adjacent_hits', 0))
        totals['global_hits'] += int(payload.get('global_hits', 0))
        totals['served_locally'] += int(payload.get('served_locally', 0))
        totals['served_from_neighbor'] += int(payload.get('served_from_neighbor', 0))
        totals['served_from_global'] += int(payload.get('served_from_global', 0))
        totals['served_from_origin'] += int(payload.get('served_from_origin', 0))
        totals['fetch_attempt_failures'] += int(payload.get('fetch_attempt_failures', payload.get('fetch_failures', 0)))
        totals['request_failures'] += int(payload.get('request_failures', _count_request_failures(payload.get('service_results', {}))))
        totals['pending_requests'] += int(payload.get('pending_requests', 0))
        per_step_hit_rates.append(float(payload.get('cache_hit_rate', 0.0)))
        service_results.update(payload.get('service_results', {}))

    replay_steps = replay.get('steps', [])
    total_requests = totals['pending_requests'] or sum(service_results.values())
    cache_hits = (
        totals['served_locally']
        + totals['served_from_neighbor']
        + totals['served_from_global']
    )
    origin_serves = totals['served_from_origin'] or service_results.get('origin_http', 0)

    summary = {
        'method': method,
        'output_dir': str(method_dir),
        'step_count': len(runtime_payloads),
        'request_count': total_requests,
        'cache_hits': cache_hits,
        'origin_serves': origin_serves,
        'fetch_failures': totals['request_failures'],
        'fetch_attempt_failures': totals['fetch_attempt_failures'],
        'request_failures': totals['request_failures'],
        'cache_hit_rate': (cache_hits / total_requests) if total_requests else 0.0,
        'origin_rate': (origin_serves / total_requests) if total_requests else 0.0,
        'avg_step_hit_rate': (sum(per_step_hit_rates) / len(per_step_hit_rates)) if per_step_hit_rates else 0.0,
        'avg_activation_seconds': (
            sum(float(step.get('activation_seconds', 0.0)) for step in replay_steps) / len(replay_steps)
        ) if replay_steps else 0.0,
        'avg_topology_seconds': (
            sum(float(step.get('topology_seconds', 0.0)) for step in replay_steps) / len(replay_steps)
        ) if replay_steps else 0.0,
        'service_results': dict(sorted(service_results.items())),
        'workload': {
            'requests_per_step': manifest.get('requests_per_step'),
            'cache_capacity': manifest.get('cache_capacity'),
            'zipf_alpha': manifest.get('zipf_alpha'),
            'migration_budget': manifest.get('migration_budget'),
            'max_satellites': manifest.get('max_satellites'),
            'max_uavs': manifest.get('max_uavs'),
            'sat_dir': manifest.get('sat_dir'),
        },
        'runtime_complete': not incomplete_steps,
        'incomplete_steps': incomplete_steps,
    }
    return summary


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# OpenSN Benchmark Summary',
        '',
        f"- steps per method: {summary['benchmark']['max_steps']}",
        f"- requests per step: {summary['benchmark']['requests_per_step']}",
        f"- request seed: {summary['benchmark']['seed']}",
        f"- cache capacity: {summary['benchmark']['cache_capacity']}",
        f"- zipf alpha: {summary['benchmark']['zipf_alpha']}",
        f"- migration budget: {summary['benchmark']['migration_budget']}",
        f"- max satellites: {summary['benchmark']['max_satellites']}",
        f"- max uavs: {summary['benchmark']['max_uavs']}",
        f"- sat dir: {summary['benchmark']['sat_dir']}",
        '',
        '| Method | Steps | Requests | Cache Hit Rate | Origin Rate | Request Failures |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method_summary in summary['methods']:
        lines.append(
            '| {method} | {steps} | {requests} | {hit:.3f} | {origin:.3f} | {failures} |'.format(
                method=method_summary['method'],
                steps=method_summary['step_count'],
                requests=method_summary['request_count'],
                hit=method_summary['cache_hit_rate'],
                origin=method_summary['origin_rate'],
                failures=method_summary['request_failures'],
            )
        )
    return '\n'.join(lines) + '\n'


def _build_summary(
    args: argparse.Namespace,
    requested_methods: list[str],
    method_summaries: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    completed_methods = [entry['method'] for entry in method_summaries]
    pending_methods = [method for method in requested_methods if method not in completed_methods]
    return {
        'benchmark': {
            'methods': requested_methods,
            'completed_methods': completed_methods,
            'pending_methods': pending_methods,
            'max_steps': args.max_steps,
            'apply_steps': args.apply_steps,
            'requests_per_step': args.requests_per_step,
            'seed': args.seed,
            'cache_capacity': args.cache_capacity,
            'zipf_alpha': args.zipf_alpha,
            'migration_budget': args.migration_budget,
            'max_satellites': args.max_satellites,
            'max_uavs': args.max_uavs,
            'sat_dir': str(args.sat_dir),
            'output_dir': str(output_root),
        },
        'methods': method_summaries,
    }


def _write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    (output_root / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_root / 'benchmark_summary.md').write_text(_build_markdown(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a realistic multi-method OpenSN benchmark and write an aggregate summary.')
    parser.add_argument('--method', default='all', help='One of otcp, nocache, lce_lru, greedy, madrl, submod, spacecache, myopic, all')
    parser.add_argument('--max-steps', type=int, default=3)
    parser.add_argument('--sat-dir', type=Path, default=Path('../traces/sat_trace_50'))
    parser.add_argument('--max-satellites', type=int, default=24)
    parser.add_argument('--max-uavs', type=int, default=6)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--satellite-image', default=DEFAULT_IMAGE)
    parser.add_argument('--ground-image', default=DEFAULT_IMAGE)
    parser.add_argument('--terminal-image', default=DEFAULT_IMAGE)
    parser.add_argument('--nano-cpu', default='50M')
    parser.add_argument('--memory-byte', default='128M')
    parser.add_argument('--apply-steps', type=int, default=0)
    parser.add_argument('--reset-each-step', action='store_true', default=True)
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--timeout', type=float, default=360.0)
    parser.add_argument('--settle-seconds', type=float, default=30.0)
    parser.add_argument('--runtime-timeout', type=float, default=120.0)
    parser.add_argument('--runtime-status-dir', type=Path, default=DEFAULT_RUNTIME_STATUS_DIR)
    parser.add_argument('--requests-per-step', type=int, default=16)
    parser.add_argument('--cache-capacity', type=int, default=8)
    parser.add_argument('--zipf-alpha', type=float, default=1.2)
    parser.add_argument('--migration-budget', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-configs', action='store_true')
    parser.add_argument('--skip-existing', dest='skip_existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    args = parser.parse_args()

    args.apply = True
    args.collect_runtime_status = True
    if args.apply_steps <= 0:
        args.apply_steps = args.max_steps

    method = _normalize_method(args.method)
    methods = list(SUPPORTED_METHODS) if method == 'all' else [method]

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    method_summaries = []
    for selected_method in methods:
        method_dir = output_root / selected_method
        expected_workload = _expected_workload(
            method=selected_method,
            requests_per_step=args.requests_per_step,
            cache_capacity=args.cache_capacity,
            zipf_alpha=args.zipf_alpha,
            migration_budget=args.migration_budget,
            max_satellites=args.max_satellites,
            max_uavs=args.max_uavs,
            sat_dir=args.sat_dir,
            seed=args.seed,
        )
        if args.skip_existing and _is_complete_method(method_dir, args.apply_steps, expected_workload):
            print(f'>>> Reusing method {selected_method} from {method_dir}', flush=True)
        else:
            if not args.skip_existing and method_dir.exists():
                shutil.rmtree(method_dir)
            run_workflow(args, selected_method, method_dir)
        if not _is_complete_method(method_dir, args.apply_steps, expected_workload):
            raise RuntimeError(
                f'Incomplete OpenSN runtime status for method {selected_method} in {method_dir}. '
                'Refusing to aggregate partial replay data.'
            )
        method_summaries.append(_aggregate_method(selected_method, method_dir))
        _write_summary(output_root, _build_summary(args, methods, method_summaries, output_root))

    summary = _build_summary(args, methods, method_summaries, output_root)
    _write_summary(output_root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
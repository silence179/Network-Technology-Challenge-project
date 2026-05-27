"""Run multi-scale OpenSN baseline replays and build aggregate artifacts.

Usage:
    python -m code.ns.opensn_scale_suite --scales 25 50 100 150
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..common.visualization import setup_font
from .opensn_benchmark import _aggregate_method, _build_markdown as _build_benchmark_markdown, _expected_workload, _is_complete_method


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = PROJECT_ROOT / 'paper'
RESULT_ROOT = PAPER_ROOT / 'results'
FIGURE_ROOT = PAPER_ROOT / 'figures'
DEFAULT_OUTPUT_ROOT = RESULT_ROOT / 'opensn_scale_baselines'
DEFAULT_FIGURE_PATH = FIGURE_ROOT / 'opensn_scale_baselines.png'
DEFAULT_METHODS = ['nocache', 'lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'otcp']
METHOD_LABELS = {
    'nocache': 'No-Cache',
    'lce_lru': 'LCE-LRU',
    'greedy': 'Greedy-Pop',
    'madrl': 'MADRL-Cache',
    'submod': 'Submod-Greedy',
    'spacecache': 'SpaceCache+',
    'myopic': 'Myopic-Opt',
    'otcp': 'OTCP',
}
METHOD_COLORS = {
    'nocache': '#95a5a6',
    'lce_lru': '#e74c3c',
    'greedy': '#e67e22',
    'madrl': '#f39c12',
    'submod': '#1abc9c',
    'spacecache': '#e91e63',
    'myopic': '#3498db',
    'otcp': '#2ecc71',
}
METHOD_MARKERS = {
    'nocache': 'o',
    'lce_lru': 's',
    'greedy': '^',
    'madrl': 'D',
    'submod': 'v',
    'spacecache': 'P',
    'myopic': 'X',
    'otcp': '*',
}


@dataclass(frozen=True)
class ScaleSpec:
    satellites: int
    trace_dir: Path


SCALE_SPECS = {
    25: ScaleSpec(25, PROJECT_ROOT / 'traces' / 'sat_trace'),
    50: ScaleSpec(50, PROJECT_ROOT / 'traces' / 'sat_trace_50'),
    100: ScaleSpec(100, PROJECT_ROOT / 'traces' / 'sat_trace_100'),
    150: ScaleSpec(150, PROJECT_ROOT / 'traces' / 'sat_trace_150'),
    200: ScaleSpec(200, PROJECT_ROOT / 'traces' / 'sat_trace_200'),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _scale_output_dir(output_root: Path, satellites: int) -> Path:
    return output_root / f'scale_{satellites}'


def _summary_path(scale_dir: Path) -> Path:
    return scale_dir / 'benchmark_summary.json'


def _method_output_dir(scale_dir: Path, method: str) -> Path:
    return scale_dir / method


def _completed_methods(
    scale_dir: Path,
    methods: list[str],
    expected_steps: int,
    *,
    requests_per_step: int,
    cache_capacity: int,
    zipf_alpha: float,
    migration_budget: int,
    max_satellites: int,
    max_uavs: int,
    sat_dir: Path,
    seed: int,
) -> list[str]:
    completed = []
    for method in methods:
        method_dir = _method_output_dir(scale_dir, method)
        expected_workload = _expected_workload(
            method=method,
            requests_per_step=requests_per_step,
            cache_capacity=cache_capacity,
            zipf_alpha=zipf_alpha,
            migration_budget=migration_budget,
            max_satellites=max_satellites,
            max_uavs=max_uavs,
            sat_dir=sat_dir,
            seed=seed,
        )
        if _is_complete_method(method_dir, expected_steps, expected_workload):
            completed.append(method)
    return completed


def _is_complete_scale(
    scale_dir: Path,
    methods: list[str],
    expected_steps: int,
    *,
    requests_per_step: int,
    cache_capacity: int,
    zipf_alpha: float,
    migration_budget: int,
    max_satellites: int,
    max_uavs: int,
    sat_dir: Path,
    seed: int,
) -> bool:
    summary_path = _summary_path(scale_dir)
    if not summary_path.exists():
        return False
    completed = set(
        _completed_methods(
            scale_dir,
            methods,
            expected_steps,
            requests_per_step=requests_per_step,
            cache_capacity=cache_capacity,
            zipf_alpha=zipf_alpha,
            migration_budget=migration_budget,
            max_satellites=max_satellites,
            max_uavs=max_uavs,
            sat_dir=sat_dir,
            seed=seed,
        )
    )
    return all(method in completed for method in methods)


def _enrich_method_summary(scale_dir: Path, method_summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(method_summary)
    replay_path = scale_dir / method_summary['method'] / 'replay_metrics.json'
    if not replay_path.exists():
        return enriched

    replay = _load_json(replay_path)
    steps = replay.get('steps', [])
    if not steps:
        return enriched

    instance_counts = [int(step.get('instance_count', 0)) for step in steps]
    link_counts = [int(step.get('link_count', 0)) for step in steps]
    start_seconds = [float(step.get('start_seconds', 0.0)) for step in steps]
    enriched['avg_instance_count'] = sum(instance_counts) / len(instance_counts)
    enriched['avg_link_count'] = sum(link_counts) / len(link_counts)
    enriched['avg_start_seconds'] = sum(start_seconds) / len(start_seconds)
    if len(set(instance_counts)) == 1:
        enriched['instance_count'] = instance_counts[0]
    if len(set(link_counts)) == 1:
        enriched['link_count'] = link_counts[0]
    return enriched


def _load_scale_result(scale_dir: Path, scale_spec: ScaleSpec) -> dict[str, Any]:
    raw_summary = _load_json(_summary_path(scale_dir))
    methods = [_enrich_method_summary(scale_dir, entry) for entry in raw_summary.get('methods', [])]
    methods_by_name = {entry['method']: entry for entry in methods}
    representative = methods[0] if methods else {}
    return {
        'scale': scale_spec.satellites,
        'sat_dir': str(scale_spec.trace_dir),
        'output_dir': str(scale_dir),
        'instance_count': representative.get('instance_count'),
        'link_count': representative.get('link_count'),
        'methods': methods,
        'methods_by_name': methods_by_name,
    }


def _benchmark_command(args: argparse.Namespace, scale_spec: ScaleSpec, scale_dir: Path) -> list[str]:
    return _benchmark_command_for_method(args, scale_spec, scale_dir, args.method)


def _benchmark_command_for_method(
    args: argparse.Namespace,
    scale_spec: ScaleSpec,
    scale_dir: Path,
    method: str,
) -> list[str]:
    command = [
        sys.executable,
        '-m',
        'code.ns.opensn_benchmark',
        '--method',
        method,
        '--max-steps',
        str(args.max_steps),
        '--apply-steps',
        str(args.apply_steps),
        '--sat-dir',
        str(scale_spec.trace_dir),
        '--max-satellites',
        str(scale_spec.satellites),
        '--max-uavs',
        str(args.max_uavs),
        '--requests-per-step',
        str(args.requests_per_step),
        '--seed',
        str(args.seed),
        '--cache-capacity',
        str(args.cache_capacity),
        '--zipf-alpha',
        str(args.zipf_alpha),
        '--migration-budget',
        str(args.migration_budget),
        '--settle-seconds',
        str(args.settle_seconds),
        '--runtime-timeout',
        str(args.runtime_timeout),
        '--timeout',
        str(args.timeout),
        '--output-dir',
        str(scale_dir),
    ]
    if args.skip_existing:
        command.append('--skip-existing')
    if args.base_url:
        command.extend(['--base-url', args.base_url])
    if args.verify_configs:
        command.append('--verify-configs')
    return command


def _build_scale_benchmark_summary(
    args: argparse.Namespace,
    scale_spec: ScaleSpec,
    scale_dir: Path,
    methods: list[str],
) -> dict[str, Any]:
    return {
        'benchmark': {
            'methods': methods,
            'max_steps': args.max_steps,
            'apply_steps': args.apply_steps,
            'requests_per_step': args.requests_per_step,
            'seed': args.seed,
            'cache_capacity': args.cache_capacity,
            'zipf_alpha': args.zipf_alpha,
            'migration_budget': args.migration_budget,
            'max_satellites': scale_spec.satellites,
            'max_uavs': args.max_uavs,
            'sat_dir': str(scale_spec.trace_dir),
            'output_dir': str(scale_dir),
        },
        'methods': [_aggregate_method(method, _method_output_dir(scale_dir, method)) for method in methods],
    }


def _write_scale_benchmark_summary(
    args: argparse.Namespace,
    scale_spec: ScaleSpec,
    scale_dir: Path,
    methods: list[str],
) -> None:
    if not methods:
        return
    summary = _build_scale_benchmark_summary(args, scale_spec, scale_dir, methods)
    _write_json(_summary_path(scale_dir), summary)
    (scale_dir / 'benchmark_summary.md').write_text(_build_benchmark_markdown(summary))


def _write_suite_outputs(
    args: argparse.Namespace,
    methods: list[str],
    scale_results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    if not scale_results:
        return

    ordered_results = sorted(scale_results, key=lambda entry: entry['scale'])
    payload = _build_suite_payload(args, methods, ordered_results)
    if failures:
        payload['failures'] = failures
    _write_json(args.output_root / 'suite_summary.json', payload)
    (args.output_root / 'suite_summary.md').write_text(_build_markdown(payload))
    _plot_suite(ordered_results, methods, args.figure_path)


def _append_scale_log(log_path: Path, header: str, result: subprocess.CompletedProcess[str]) -> None:
    parts = [f'>>> {header}']
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(result.stderr.rstrip())
    parts.append('')
    with log_path.open('a') as handle:
        handle.write('\n'.join(parts))


def _run_scale(args: argparse.Namespace, scale_spec: ScaleSpec, methods: list[str]) -> dict[str, Any]:
    scale_dir = _scale_output_dir(args.output_root, scale_spec.satellites)
    summary_path = _summary_path(scale_dir)
    log_path = scale_dir / 'suite_run.log'
    required_steps = max(args.apply_steps, 1)

    completion_kwargs = {
        'requests_per_step': args.requests_per_step,
        'cache_capacity': args.cache_capacity,
        'zipf_alpha': args.zipf_alpha,
        'migration_budget': args.migration_budget,
        'max_satellites': scale_spec.satellites,
        'max_uavs': args.max_uavs,
        'sat_dir': scale_spec.trace_dir,
        'seed': args.seed,
    }

    if args.skip_existing and _is_complete_scale(scale_dir, methods, required_steps, **completion_kwargs):
        _write_scale_benchmark_summary(args, scale_spec, scale_dir, methods)
        print(f'>>> Reusing scale {scale_spec.satellites} from {scale_dir}', flush=True)
        return _load_scale_result(scale_dir, scale_spec)

    if args.skip_run:
        if not summary_path.exists():
            raise FileNotFoundError(f'Missing benchmark summary: {summary_path}')
        return _load_scale_result(scale_dir, scale_spec)

    if scale_dir.exists() and not args.skip_existing:
        shutil.rmtree(scale_dir)
    scale_dir.mkdir(parents=True, exist_ok=True)

    completed_methods = _completed_methods(scale_dir, methods, required_steps, **completion_kwargs) if args.skip_existing else []
    pending_methods = [method for method in methods if method not in completed_methods]
    if completed_methods:
        print(
            f'>>> Resuming scale {scale_spec.satellites} with completed methods: {", ".join(completed_methods)}',
            flush=True,
        )
        _write_scale_benchmark_summary(args, scale_spec, scale_dir, completed_methods)

    if not pending_methods:
        _write_scale_benchmark_summary(args, scale_spec, scale_dir, methods)
        print(f'>>> Reusing scale {scale_spec.satellites} from {scale_dir}', flush=True)
        return _load_scale_result(scale_dir, scale_spec)

    print(f'>>> Running OpenSN suite at {scale_spec.satellites} satellites', flush=True)
    for method in pending_methods:
        method_dir = _method_output_dir(scale_dir, method)
        if method_dir.exists() and not args.skip_existing:
            shutil.rmtree(method_dir)
        command = _benchmark_command_for_method(args, scale_spec, scale_dir, method)
        result = subprocess.run(
            command,
            cwd=PAPER_ROOT,
            capture_output=True,
            text=True,
        )
        _append_scale_log(log_path, f'Method {method}', result)
        if result.returncode != 0:
            raise RuntimeError(
                f'Scale {scale_spec.satellites} failed while running {method} '
                f'with exit code {result.returncode}. See {log_path}.'
            )
        expected_workload = _expected_workload(
            method=method,
            requests_per_step=args.requests_per_step,
            cache_capacity=args.cache_capacity,
            zipf_alpha=args.zipf_alpha,
            migration_budget=args.migration_budget,
            max_satellites=scale_spec.satellites,
            max_uavs=args.max_uavs,
            sat_dir=scale_spec.trace_dir,
            seed=args.seed,
        )
        if not _is_complete_method(method_dir, args.apply_steps, expected_workload):
            raise RuntimeError(
                f'Scale {scale_spec.satellites} produced incomplete output for {method}. '
                f'See {method_dir}.'
            )
        completed_methods.append(method)
        _write_scale_benchmark_summary(args, scale_spec, scale_dir, completed_methods)
        print(
            f'>>> Checkpoint scale {scale_spec.satellites}: saved {method} ({len(completed_methods)}/{len(methods)})',
            flush=True,
        )

    scale_result = _load_scale_result(scale_dir, scale_spec)
    best_method = max(scale_result['methods'], key=lambda entry: entry.get('cache_hit_rate', 0.0))
    otcp = scale_result['methods_by_name'].get('otcp', {})
    print(
        '>>> Completed {scale} satellites: best={best} ({best_hit:.1f}%), OTCP={otcp_hit:.1f}%'.format(
            scale=scale_spec.satellites,
            best=METHOD_LABELS.get(best_method['method'], best_method['method']),
            best_hit=100.0 * float(best_method.get('cache_hit_rate', 0.0)),
            otcp_hit=100.0 * float(otcp.get('cache_hit_rate', 0.0)),
        ),
        flush=True,
    )
    return scale_result


def _build_suite_payload(args: argparse.Namespace, methods: list[str], scale_results: list[dict[str, Any]]) -> dict[str, Any]:
    cleaned_results = []
    for scale_result in scale_results:
        cleaned_results.append(
            {
                key: value
                for key, value in scale_result.items()
                if key != 'methods_by_name'
            }
        )
    return {
        'benchmark': {
            'methods': methods,
            'scales': [entry['scale'] for entry in cleaned_results],
            'max_steps': args.max_steps,
            'apply_steps': args.apply_steps,
            'requests_per_step': args.requests_per_step,
            'seed': args.seed,
            'cache_capacity': args.cache_capacity,
            'zipf_alpha': args.zipf_alpha,
            'migration_budget': args.migration_budget,
            'max_uavs': args.max_uavs,
            'settle_seconds': args.settle_seconds,
            'runtime_timeout': args.runtime_timeout,
            'timeout': args.timeout,
            'base_url': args.base_url,
            'output_root': str(args.output_root),
            'figure_path': str(args.figure_path),
        },
        'scales': cleaned_results,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# OpenSN Multi-Scale Baseline Summary',
        '',
        f"- scales: {', '.join(str(scale) for scale in payload['benchmark']['scales'])}",
        f"- requests per step: {payload['benchmark']['requests_per_step']}",
        f"- request seed: {payload['benchmark']['seed']}",
        f"- cache capacity: {payload['benchmark']['cache_capacity']}",
        f"- zipf alpha: {payload['benchmark']['zipf_alpha']}",
        f"- migration budget: {payload['benchmark']['migration_budget']}",
        f"- max uavs: {payload['benchmark']['max_uavs']}",
        '',
        '| Scale | Best Hit Method | Best Hit Rate | OTCP Hit Rate | Activation Range (s) | Instances | Links |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for scale_entry in payload['scales']:
        methods = scale_entry['methods']
        best = max(methods, key=lambda entry: entry.get('cache_hit_rate', 0.0))
        otcp = next((entry for entry in methods if entry['method'] == 'otcp'), None)
        activations = [float(entry.get('avg_activation_seconds', 0.0)) for entry in methods]
        activation_range = f"{min(activations):.3f}-{max(activations):.3f}"
        otcp_hit = 'n/a' if otcp is None else f"{100.0 * float(otcp.get('cache_hit_rate', 0.0)):.1f}%"
        lines.append(
            '| {scale} | {best} | {best_hit:.1f}% | {otcp_hit} | {activation_range} | {instances} | {links} |'.format(
                scale=scale_entry['scale'],
                best=METHOD_LABELS.get(best['method'], best['method']),
                best_hit=100.0 * float(best.get('cache_hit_rate', 0.0)),
                otcp_hit=otcp_hit,
                activation_range=activation_range,
                instances=scale_entry.get('instance_count', 'n/a'),
                links=scale_entry.get('link_count', 'n/a'),
            )
        )

    lines.extend(
        [
            '',
            '| Scale | Method | Hit Rate | Origin Rate | Failures | Topology (s) | Activation (s) | Instances | Links |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for scale_entry in payload['scales']:
        for method_entry in scale_entry['methods']:
            lines.append(
                '| {scale} | {method} | {hit:.1f}% | {origin:.1f}% | {failures} | {topology:.3f} | {activation:.3f} | {instances} | {links} |'.format(
                    scale=scale_entry['scale'],
                    method=METHOD_LABELS.get(method_entry['method'], method_entry['method']),
                    hit=100.0 * float(method_entry.get('cache_hit_rate', 0.0)),
                    origin=100.0 * float(method_entry.get('origin_rate', 0.0)),
                    failures=int(method_entry.get('request_failures', method_entry.get('fetch_failures', 0))),
                    topology=float(method_entry.get('avg_topology_seconds', 0.0)),
                    activation=float(method_entry.get('avg_activation_seconds', 0.0)),
                    instances=method_entry.get('instance_count', scale_entry.get('instance_count', 'n/a')),
                    links=method_entry.get('link_count', scale_entry.get('link_count', 'n/a')),
                )
            )
    lines.append('')
    return '\n'.join(lines)


def _plot_metric(ax, scale_results: list[dict[str, Any]], methods: list[str], key: str, title: str, ylabel: str, scale: str = 'linear') -> None:
    xs = [entry['scale'] for entry in scale_results]
    for method in methods:
        ys = []
        for scale_result in scale_results:
            method_summary = scale_result['methods_by_name'][method]
            value = float(method_summary.get(key, 0.0))
            if key in {'cache_hit_rate', 'origin_rate'}:
                value *= 100.0
            ys.append(max(value, 1e-3) if scale == 'log' else value)
        ax.plot(
            xs,
            ys,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.0,
            markersize=6,
            label=METHOD_LABELS[method],
        )
    ax.set_title(title)
    ax.set_xlabel('Visible Satellites')
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    if scale == 'log':
        ax.set_yscale('log')


def _plot_suite(scale_results: list[dict[str, Any]], methods: list[str], figure_path: Path) -> None:
    setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle('OpenSN Multi-Scale Replay Across Baselines', fontsize=13, fontweight='bold')

    _plot_metric(axes[0], scale_results, methods, 'cache_hit_rate', 'Successful Cache Serve Rate', 'Successful Cache Serve (%)')
    _plot_metric(axes[1], scale_results, methods, 'origin_rate', 'Successful Origin Serve Rate', 'Successful Origin Serve (%)')
    _plot_metric(axes[2], scale_results, methods, 'avg_activation_seconds', 'Activation Latency', 'Activation (s)', scale='log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run all-method OpenSN replays across multiple constellation scales and build aggregate summaries.')
    parser.add_argument('--method', default='all')
    parser.add_argument('--scales', nargs='*', type=int, default=[25, 50, 100, 150])
    parser.add_argument('--max-steps', type=int, default=3)
    parser.add_argument('--apply-steps', type=int, default=3)
    parser.add_argument('--max-uavs', type=int, default=6)
    parser.add_argument('--requests-per-step', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-capacity', type=int, default=8)
    parser.add_argument('--zipf-alpha', type=float, default=1.2)
    parser.add_argument('--migration-budget', type=int, default=4)
    parser.add_argument('--settle-seconds', type=float, default=30.0)
    parser.add_argument('--runtime-timeout', type=float, default=120.0)
    parser.add_argument('--timeout', type=float, default=900.0)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--figure-path', type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument('--skip-existing', dest='skip_existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    parser.add_argument('--skip-run', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--verify-configs', action='store_true')
    args = parser.parse_args()

    if args.method != 'all':
        methods = [args.method]
    else:
        methods = list(DEFAULT_METHODS)

    args.output_root.mkdir(parents=True, exist_ok=True)
    scale_specs = []
    for scale in args.scales:
        if scale not in SCALE_SPECS:
            raise ValueError(f'Unsupported scale: {scale}')
        scale_specs.append(SCALE_SPECS[scale])

    scale_results = []
    failures = []
    for scale_spec in scale_specs:
        try:
            scale_result = _run_scale(args, scale_spec, methods)
            scale_results.append(scale_result)
            _write_suite_outputs(args, methods, scale_results, failures)
        except Exception as exc:
            failures.append({'scale': scale_spec.satellites, 'error': str(exc)})
            print(f'>>> Scale {scale_spec.satellites} failed: {exc}', flush=True)
            _write_suite_outputs(args, methods, scale_results, failures)
            if args.fail_fast:
                raise

    if not scale_results:
        raise RuntimeError('No scale completed successfully.')

    scale_results.sort(key=lambda entry: entry['scale'])
    payload = _build_suite_payload(args, methods, scale_results)
    if failures:
        payload['failures'] = failures
    _write_json(args.output_root / 'suite_summary.json', payload)
    (args.output_root / 'suite_summary.md').write_text(_build_markdown(payload))
    _plot_suite(scale_results, methods, args.figure_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
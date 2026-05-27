"""Reproduction-oriented SpaceCache+ experiment with per-step coverage refresh.

Usage:
    python -m code.repro.spacecache_experiment
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..baselines.spacecache_plus import (
    _build_overlap_graph,
    _compute_coverage,
    route_spacecache,
    spacecache_placement,
)
from ..common.metrics import generate_requests
from ..common.popularity import PopularityTracker
from .common import build_snapshot, compute_basic_metrics, count_unique_cached, load_sampled_traces, seed_everything


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'results' / 'spacecache_repro_metrics.json'


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduction-oriented SpaceCache+ experiment.')
    parser.add_argument('--max-steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--redundancy-penalty', type=float, default=0.15)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    seed_everything(args.seed)
    df_sat, df_uav, timestamps = load_sampled_traces(max_steps=args.max_steps)
    if not timestamps:
        raise RuntimeError('No sampled timestamps available for SpaceCache+ reproduction.')

    tracker = PopularityTracker()
    delays = []
    traffics = []
    diversity = []
    hits = 0
    total = 0
    solve_time_ms = 0.0
    candidate_counts = []
    coverage_avgs = []
    overlap_degrees = []
    t0 = time.time()

    for t_ms in timestamps:
        snapshot = build_snapshot(df_sat, df_uav, t_ms)
        if snapshot is None:
            continue

        pop_scores = dict(tracker.scores)
        step_solve_start = time.time()
        placement = spacecache_placement(
            snapshot.G,
            snapshot.all_reachable,
            snapshot.type_map,
            pop_scores,
            redundancy_penalty=args.redundancy_penalty,
        )
        solve_time_ms += (time.time() - step_solve_start) * 1000.0
        diversity.append(count_unique_cached(placement))
        candidate_counts.append(len(snapshot.all_reachable))

        coverage_scores, user_sets = _compute_coverage(snapshot.G, snapshot.all_reachable, snapshot.type_map)
        if coverage_scores:
            coverage_avgs.append(float(np.mean(list(coverage_scores.values()))))
        overlap_graph = _build_overlap_graph(snapshot.all_reachable, user_sets)
        if overlap_graph:
            overlap_degrees.append(float(np.mean([len(v) for v in overlap_graph.values()])))

        requests = generate_requests(snapshot.G, snapshot.type_map)
        if not requests:
            continue
        tracker.decay_all()

        for requester, content_id in requests:
            tracker.record(content_id)
            delay, traffic, hit = route_spacecache(
                snapshot.G,
                requester,
                placement,
                content_id,
                snapshot.type_map,
            )
            if delay is None:
                continue
            delays.append(delay)
            traffics.append(traffic)
            hits += int(hit)
            total += 1

    payload = {
        'method': 'SpaceCache+',
        'reference': 'Fang et al., IEEE INFOCOM 2024',
        'max_steps': len(timestamps),
        'elapsed_seconds': time.time() - t0,
        'evaluation_summary': compute_basic_metrics(
            delays,
            traffics,
            hits,
            total,
            diversity,
            extra={
                'solve_time_ms': solve_time_ms,
                'avg_candidate_sats': float(np.mean(candidate_counts)) if candidate_counts else 0.0,
                'avg_coverage_score': float(np.mean(coverage_avgs)) if coverage_avgs else 0.0,
                'avg_overlap_degree': float(np.mean(overlap_degrees)) if overlap_degrees else 0.0,
                'redundancy_penalty': args.redundancy_penalty,
                'refresh_interval_steps': 1,
            },
        ),
    }

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
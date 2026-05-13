"""Reproduction-oriented MADRL experiment with explicit train/eval split.

Usage:
    python -m code.repro.madrl_experiment
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..baselines.drl_actor_critic import MaDRLManager, route_madrl
from ..common.metrics import generate_requests
from ..common.popularity import PopularityTracker
from .common import (
    best_serving_node,
    build_snapshot,
    compute_basic_metrics,
    count_unique_cached,
    load_sampled_traces,
    seed_everything,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'results' / 'madrl_repro_metrics.json'


def _roll_step(snapshot, tracker, manager, *, training: bool):
    pop_scores = dict(tracker.scores)
    placement = manager.decide_placement(snapshot.all_reachable, pop_scores)
    diversity = count_unique_cached(placement)

    requests = generate_requests(snapshot.G, snapshot.type_map)
    if not requests:
        return {
            'delays': [],
            'traffics': [],
            'hits': 0,
            'total': 0,
            'diversity': diversity,
            'reward': 0.0,
        }

    tracker.decay_all()
    delays = []
    traffics = []
    hits = 0
    total = 0
    hit_counts = {node: 0 for node in snapshot.all_reachable}

    for requester, content_id in requests:
        tracker.record(content_id)
        serving_node = best_serving_node(snapshot.G, requester, placement, content_id)
        delay, traffic, hit = route_madrl(
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
        if hit and serving_node is not None:
            hit_counts[serving_node] = hit_counts.get(serving_node, 0) + 1

    if training:
        manager.feedback(snapshot.all_reachable, hit_counts, pop_scores)

    return {
        'delays': delays,
        'traffics': traffics,
        'hits': hits,
        'total': total,
        'diversity': diversity,
        'reward': float(sum(hit_counts.values())),
    }


def _run_training(df_sat, df_uav, timestamps, train_steps: int, train_episodes: int, seed: int):
    manager = MaDRLManager()
    episode_rewards = []
    episode_hit_rates = []

    for episode in range(train_episodes):
        seed_everything(seed + episode)
        tracker = PopularityTracker()
        reward_sum = 0.0
        hits = 0
        total = 0

        for t_ms in timestamps[:train_steps]:
            snapshot = build_snapshot(df_sat, df_uav, t_ms)
            if snapshot is None:
                continue
            step_result = _roll_step(snapshot, tracker, manager, training=True)
            reward_sum += step_result['reward']
            hits += step_result['hits']
            total += step_result['total']

        episode_rewards.append(reward_sum)
        episode_hit_rates.append(hits / total if total else 0.0)

    return manager, episode_rewards, episode_hit_rates


def _warmstart_tracker(df_sat, df_uav, timestamps, train_steps: int):
    tracker = PopularityTracker()
    for t_ms in timestamps[:train_steps]:
        snapshot = build_snapshot(df_sat, df_uav, t_ms)
        if snapshot is None:
            continue
        requests = generate_requests(snapshot.G, snapshot.type_map)
        if not requests:
            continue
        tracker.decay_all()
        for _, content_id in requests:
            tracker.record(content_id)
    return tracker


def _run_evaluation(df_sat, df_uav, timestamps, train_steps: int, manager, seed: int):
    seed_everything(seed)
    tracker = _warmstart_tracker(df_sat, df_uav, timestamps, train_steps)
    manager.set_eval_mode()

    delays = []
    traffics = []
    diversity = []
    hits = 0
    total = 0

    for t_ms in timestamps[train_steps:]:
        snapshot = build_snapshot(df_sat, df_uav, t_ms)
        if snapshot is None:
            continue
        step_result = _roll_step(snapshot, tracker, manager, training=False)
        delays.extend(step_result['delays'])
        traffics.extend(step_result['traffics'])
        diversity.append(step_result['diversity'])
        hits += step_result['hits']
        total += step_result['total']

    return compute_basic_metrics(
        delays,
        traffics,
        hits,
        total,
        diversity,
        extra={'trained_agents': manager.agent_count()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduction-oriented MADRL experiment.')
    parser.add_argument('--max-steps', type=int, default=120)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--train-episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df_sat, df_uav, timestamps = load_sampled_traces(max_steps=args.max_steps)
    if not timestamps:
        raise RuntimeError('No sampled timestamps available for MADRL reproduction.')

    train_steps = max(1, min(int(len(timestamps) * args.train_ratio), len(timestamps) - 1))

    t0 = time.time()
    manager, episode_rewards, episode_hit_rates = _run_training(
        df_sat,
        df_uav,
        timestamps,
        train_steps,
        args.train_episodes,
        args.seed,
    )
    eval_metrics = _run_evaluation(
        df_sat,
        df_uav,
        timestamps,
        train_steps,
        manager,
        args.seed + 1000,
    )

    payload = {
        'method': 'MADRL-Cache',
        'reference': 'Zhong et al., IEEE TCCN 2020',
        'max_steps': len(timestamps),
        'train_steps': train_steps,
        'eval_steps': len(timestamps) - train_steps,
        'train_episodes': args.train_episodes,
        'elapsed_seconds': time.time() - t0,
        'training_summary': {
            'episode_rewards': episode_rewards,
            'episode_hit_rates': episode_hit_rates,
            'mean_episode_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_hit_rate': sum(episode_hit_rates) / len(episode_hit_rates) if episode_hit_rates else 0.0,
        },
        'evaluation_summary': eval_metrics,
    }

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
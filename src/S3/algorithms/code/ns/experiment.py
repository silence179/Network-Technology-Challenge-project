"""
Experiment runner using SimPy discrete-event network simulator.

Reuses the existing topology, caching algorithms, and trace infrastructure,
but routes every content request through the SimPy DES engine for genuine
stochastic simulation with ARQ retransmission, queuing, and packet loss.

Usage:
    python -m code.ns.experiment [--mode {main,ablation,zipf,capacity,all}]
"""

import sys
import os
import json
import argparse
import time as _time
import numpy as np
import networkx as nx
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reuse existing infrastructure ──
from ..config import (
    SAT_DIR, MAX_STEPS, REQUESTS_PER_STEP,
    CONTENT_SIZE_MB, CONTENT_SIZE_BITS,
    GREEDY_REFRESH_INTERVAL, STEP_STRIDE,
    CONTENT_CATALOG_SIZE,
    CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS,
)
from .. import config as cfg
from ..common.data_loader import load_traces, get_nodes
from ..common.topology import build_topology, get_cache_nodes, get_all_reachable_sats
from ..common.metrics import generate_requests
from ..common.popularity import PopularityTracker
from ..olcp.solver import solve_olcp
from ..baselines.no_cache import route_nocache
from ..baselines.lce_lru import LCELRUManager
from ..baselines.greedy_popular import greedy_placement, route_greedy
from ..baselines.myopic import solve_myopic, route_myopic
from ..baselines.drl_actor_critic import MaDRLManager, route_madrl
from ..baselines.spacecache_plus import spacecache_placement, route_spacecache
from ..baselines.submodular_greedy import submodular_greedy_placement, route_submodular

from .simulator import NetworkSimulator, FlowResult

CONTENT_SIZE_BYTES = int(CONTENT_SIZE_MB * 1e6)

METHODS = ['nocache', 'lce_lru', 'greedy', 'madrl', 'submod',
           'spacecache', 'myopic', 'olcp']

METHOD_LABELS = {
    'nocache': 'No-Cache', 'lce_lru': 'LCE-LRU', 'greedy': 'Greedy-Pop',
    'madrl': 'MADRL-Cache', 'submod': 'Submod-Greedy',
    'spacecache': 'SpaceCache+', 'myopic': 'Myopic-Opt', 'olcp': 'OTCP (Ours)',
}


# ── Helper: find path and determine cache hit ──

def find_serving_path(G, requester, placement, content_id, type_map):
    """Find the best serving path for a content request.

    Returns (path, is_cache_hit, serving_node).
    - Cache hit: shortest path to any cache node holding content_id.
    - Cache miss: shortest path to ground station (origin).
    """
    if not G.has_node(requester):
        return None, False, None

    # Stage 1: check cache
    best_delay = float('inf')
    best_path = None
    best_node = None
    for node, cached_items in placement.items():
        if content_id not in cached_items:
            continue
        if not G.has_node(node):
            continue
        try:
            path = nx.dijkstra_path(G, requester, node, weight='eff_delay')
            d = sum(G[path[i]][path[i+1]]['eff_delay'] for i in range(len(path)-1))
            if d < best_delay:
                best_delay = d
                best_path = path
                best_node = node
        except nx.NetworkXNoPath:
            pass

    if best_path is not None:
        return best_path, True, best_node

    # Stage 2: origin fallback
    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    best_delay = float('inf')
    best_path = None
    best_node = None
    for gs in gs_nodes:
        try:
            path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
            d = sum(G[path[i]][path[i+1]]['eff_delay'] for i in range(len(path)-1))
            if d < best_delay:
                best_delay = d
                best_path = path
                best_node = gs
        except nx.NetworkXNoPath:
            pass

    return best_path, False, best_node


def _count_unique_cached(placement):
    all_items = set()
    for contents in placement.values():
        all_items.update(contents)
    return len(all_items)


def _build_future_snapshots(df_sat, df_uav, timestamps, start_idx, horizon):
    snapshots = []
    for i in range(start_idx, min(start_idx + horizon + 1, len(timestamps))):
        t_ms = timestamps[i]
        nodes_df = get_nodes(df_sat, df_uav, int(t_ms))
        if nodes_df.empty:
            continue
        G, coord_map, type_map = build_topology(nodes_df)
        if len(G.nodes) < 2:
            continue
        cache_nodes = get_cache_nodes(G, type_map)
        all_reachable = get_all_reachable_sats(G, type_map)
        snapshots.append({
            'G': G, 'cache_nodes': cache_nodes, 'type_map': type_map,
            'coord_map': coord_map, 'all_reachable': all_reachable,
        })
    return snapshots


# ── Main experiment ──

def run_ns_experiment(sat_dir=SAT_DIR, max_steps=MAX_STEPS):
    """Run the full experiment using SimPy network simulation."""
    np.random.seed(42)
    random.seed(42)

    print(">>> [NS] Loading traces...")
    df_sat, df_uav, timestamps = load_traces(sat_dir)
    if not timestamps:
        print("[ERROR] No timestamps.")
        return None
    timestamps = timestamps[::STEP_STRIDE][:max_steps]
    H = cfg.LOOKAHEAD_HORIZON
    n_steps = len(timestamps)
    print(f"    SAT files: {len(os.listdir(sat_dir))}, time steps: {len(timestamps)*STEP_STRIDE}")
    print(f">>> [NS] SimPy experiment: {n_steps} steps, H={H}")

    # Per-method accumulators
    method_flows = {m: [] for m in METHODS}
    method_solve_ms = {m: 0.0 for m in METHODS}
    method_diversity = {m: [] for m in METHODS}

    tracker = PopularityTracker()
    lce_mgr = LCELRUManager()
    madrl_mgr = MaDRLManager()

    # Cache states
    greedy_state, madrl_state, submod_state = {}, {}, {}
    spacecache_state, myopic_state, olcp_state = {}, {}, {}

    t0 = _time.time()

    for step in range(n_steps):
        if step % 10 == 0:
            print(f"  Step {step}/{n_steps} ({_time.time()-t0:.1f}s)", flush=True)

        snapshots = _build_future_snapshots(df_sat, df_uav, timestamps, step, H)
        if not snapshots:
            continue

        cur = snapshots[0]
        G = cur['G']
        type_map = cur['type_map']
        cache_nodes = cur['cache_nodes']
        if not cache_nodes:
            continue
        all_reachable = cur.get('all_reachable', cache_nodes)

        pop_scores = dict(tracker.scores)

        # ── Solve placements (same as original experiment) ──
        olcp_snaps = [{'G': s['G'],
                        'cache_nodes': s.get('all_reachable', s['cache_nodes']),
                        'type_map': s['type_map']} for s in snapshots]
        t_s = _time.time()
        olcp_state, _ = solve_olcp(olcp_snaps, olcp_state, pop_scores)
        method_solve_ms['olcp'] += (_time.time() - t_s) * 1000

        myopic_snap = {'G': G,
                       'cache_nodes': cur.get('all_reachable', cache_nodes),
                       'type_map': type_map}
        t_s = _time.time()
        myopic_state, _ = solve_myopic(myopic_snap, myopic_state, pop_scores)
        method_solve_ms['myopic'] += (_time.time() - t_s) * 1000

        if step % GREEDY_REFRESH_INTERVAL == 0:
            greedy_state = greedy_placement(cache_nodes, tracker)
        madrl_state = madrl_mgr.decide_placement(all_reachable, pop_scores)
        if step % GREEDY_REFRESH_INTERVAL == 0:
            t_s = _time.time()
            submod_state = submodular_greedy_placement(G, all_reachable, type_map, pop_scores)
            method_solve_ms['submod'] += (_time.time() - t_s) * 1000
        if step % GREEDY_REFRESH_INTERVAL == 0:
            t_s = _time.time()
            spacecache_state = spacecache_placement(G, all_reachable, type_map, pop_scores)
            method_solve_ms['spacecache'] += (_time.time() - t_s) * 1000

        # Record diversity
        method_diversity['greedy'].append(_count_unique_cached(greedy_state))
        method_diversity['madrl'].append(_count_unique_cached(madrl_state))
        method_diversity['submod'].append(_count_unique_cached(submod_state))
        method_diversity['spacecache'].append(_count_unique_cached(spacecache_state))
        method_diversity['myopic'].append(_count_unique_cached(myopic_state))
        method_diversity['olcp'].append(_count_unique_cached(olcp_state))

        # ── Generate requests ──
        requests = generate_requests(G, type_map)
        if not requests:
            continue
        tracker.decay_all()
        for _, cid in requests:
            tracker.record(cid)

        # ── For each method: find path, then simulate via SimPy ──
        # Extract LCE-LRU state from internal stores
        lce_state = {}
        for node, store in lce_mgr._stores.items():
            lce_state[node] = store.contents()

        placements = {
            'nocache': {},
            'lce_lru': lce_state,
            'greedy': greedy_state,
            'madrl': madrl_state,
            'submod': submod_state,
            'spacecache': spacecache_state,
            'myopic': myopic_state,
            'olcp': olcp_state,
        }
        madrl_hit_counts = {c: 0 for c in all_reachable}

        for method in METHODS:
            # Create a fresh simulator per method per step
            ns = NetworkSimulator(seed=42 + step * 100 + METHODS.index(method))
            ns.build_from_graph(G)

            placement = placements[method]

            for requester, cid in requests:
                path, is_hit, serving_node = find_serving_path(
                    G, requester, placement, cid, type_map)

                if path is None:
                    # No path available → origin fallback attempt
                    gs_nodes = [n for n, t in type_map.items()
                                if t == 'GS' and G.has_node(n)]
                    for gs in gs_nodes:
                        try:
                            path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
                            serving_node = gs
                            break
                        except nx.NetworkXNoPath:
                            path = None
                    is_hit = False

                if path is None:
                    continue

                if method == 'madrl' and is_hit and serving_node is not None:
                    madrl_hit_counts[serving_node] = madrl_hit_counts.get(serving_node, 0) + 1

                ns.submit_flow(path, cid, CONTENT_SIZE_BYTES, is_hit)

            # Run simulation until all flows complete
            ns.run_until_done()

            # Collect results
            flows = ns.collect_results()
            method_flows[method].extend(flows)

        # LCE-LRU: also update its internal cache state
        for requester, cid in requests:
            lce_mgr.route(G, requester, cache_nodes, cid, type_map)

        # MADRL feedback
        madrl_mgr.feedback(all_reachable, madrl_hit_counts, pop_scores)

    elapsed = _time.time() - t0
    print(f">>> [NS] Completed in {elapsed:.1f}s")

    # ── Compute metrics ──
    return _compute_all_metrics(method_flows, method_solve_ms, method_diversity, n_steps)


def _compute_all_metrics(method_flows, method_solve_ms, method_diversity, n_steps):
    """Compute final metrics from all SimPy flow results."""
    metrics = {}
    for m in METHODS:
        flows = method_flows[m]
        total = len(flows)
        delivered = [f for f in flows if f.delivered]
        delays = [f.delay_ms for f in delivered]
        hits = sum(1 for f in delivered if f.cache_hit)
        retrans = [f.retransmissions for f in flows]
        traffic_mb = sum(f.hops * CONTENT_SIZE_MB *
                         (2 if not f.cache_hit else 1)
                         for f in delivered)

        div_list = method_diversity.get(m, [])

        metrics[m] = {
            'avg_delay_ms': float(np.mean(delays)) if delays else 0,
            'median_delay_ms': float(np.median(delays)) if delays else 0,
            'std_delay_ms': float(np.std(delays)) if delays else 0,
            'p95_delay_ms': float(np.percentile(delays, 95)) if len(delays) >= 2 else 0,
            'p99_delay_ms': float(np.percentile(delays, 99)) if len(delays) >= 2 else 0,
            'total_traffic_gb': traffic_mb / 1024.0,
            'hit_rate': hits / total if total > 0 else 0,
            'backhaul_rate': 1.0 - (hits / total) if total > 0 else 1.0,
            'delivery_rate': len(delivered) / total if total > 0 else 0,
            'dropped': total - len(delivered),
            'solve_time_ms': method_solve_ms.get(m, 0),
            'avg_diversity': float(np.mean(div_list)) if div_list else 0,
            'avg_retransmissions': float(np.mean(retrans)) if retrans else 0,
            'total_retransmissions': int(sum(retrans)),
            'total_requests': total,
        }

    return metrics


# ── Display ──

def print_results(metrics):
    hdr = f"{'Method':<20s} {'Delay(ms)':>10s} {'Std':>8s} {'P95':>10s} " \
          f"{'Hit Rate':>10s} {'Backhaul':>10s} {'Diver':>7s} " \
          f"{'Retrans':>8s} {'Drop':>5s} {'Solve(ms)':>10s}"
    sep = '-' * len(hdr)
    print('\n' + '=' * len(hdr))
    print('  OTCP Network Simulation Results (SimPy DES)')
    print('=' * len(hdr))
    print(hdr)
    print(sep)
    for m in METHODS:
        d = metrics[m]
        print(f"{METHOD_LABELS.get(m, m):<20s} "
              f"{d['avg_delay_ms']:>10.1f} "
              f"{d.get('std_delay_ms', 0):>8.1f} "
              f"{d.get('p95_delay_ms', 0):>10.1f} "
              f"{d['hit_rate']:>9.1%} "
              f"{d['backhaul_rate']:>9.1%} "
              f"{d['avg_diversity']:>7.1f} "
              f"{d.get('avg_retransmissions', 0):>8.2f} "
              f"{d.get('dropped', 0):>5d} "
              f"{d['solve_time_ms']:>10.0f}")
    print(sep)

    # OTCP vs baselines
    if metrics.get('olcp') and metrics.get('nocache'):
        otcp_d = metrics['olcp']['avg_delay_ms']
        nc_d = metrics['nocache']['avg_delay_ms']
        if nc_d > 0:
            print(f"\n  OTCP vs No-Cache: delay reduction {(nc_d-otcp_d)/nc_d*100:.1f}%")
    print()


# ── Save ──

def save_results(metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'ns_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f">>> [NS] Metrics saved: {path}")


# ── Figures ──

def plot_main_results(metrics, fig_dir):
    """Generate comparison bar charts."""
    os.makedirs(fig_dir, exist_ok=True)
    methods = METHODS
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = ['#95a5a6', '#e74c3c', '#e67e22', '#f39c12',
              '#1abc9c', '#e91e63', '#3498db', '#2ecc71']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Delay
    vals = [metrics[m]['avg_delay_ms'] for m in methods]
    axes[0, 0].bar(labels, vals, color=colors)
    axes[0, 0].set_ylabel('Avg Delay (ms)')
    axes[0, 0].set_title('Average Content Delivery Delay')
    axes[0, 0].tick_params(axis='x', rotation=30)

    # Hit rate
    vals = [metrics[m]['hit_rate'] * 100 for m in methods]
    axes[0, 1].bar(labels, vals, color=colors)
    axes[0, 1].set_ylabel('Hit Rate (%)')
    axes[0, 1].set_title('Cache Hit Rate')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # Retransmissions
    vals = [metrics[m].get('avg_retransmissions', 0) for m in methods]
    axes[1, 0].bar(labels, vals, color=colors)
    axes[1, 0].set_ylabel('Avg Retransmissions')
    axes[1, 0].set_title('ARQ Retransmissions per Request')
    axes[1, 0].tick_params(axis='x', rotation=30)

    # Delivery rate
    vals = [metrics[m].get('delivery_rate', 1.0) * 100 for m in methods]
    axes[1, 1].bar(labels, vals, color=colors)
    axes[1, 1].set_ylabel('Delivery Rate (%)')
    axes[1, 1].set_title('Successful Delivery Rate')
    axes[1, 1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(fig_dir, 'ns_experiment_results.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f">>> [NS] Figure saved: {path}")


# ── Entry point ──

def main():
    parser = argparse.ArgumentParser(description='OTCP SimPy Network Simulator Experiments')
    parser.add_argument('--mode', default='main',
                        choices=['main', 'all'],
                        help='Experiment mode')
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(cfg.CODE_DIR), 'results')
    fig_dir = os.path.join(os.path.dirname(cfg.CODE_DIR), 'figures')
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 60)
    print("  OTCP: SimPy Discrete Event Network Simulation")
    print("=" * 60)

    metrics = run_ns_experiment()
    if metrics:
        print_results(metrics)
        save_results(metrics, base_dir)
        plot_main_results(metrics, fig_dir)


if __name__ == '__main__':
    main()

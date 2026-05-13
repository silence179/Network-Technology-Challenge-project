"""
OLCP Main Experiment: Orbit-Lookahead Content Placement vs Baselines.

Compares 8 schemes:
  1. No-Cache        — pure Dijkstra to origin
  2. LCE-LRU         — classic NDN (Leave Copy Everywhere + LRU)
  3. Greedy-Popular   — place globally popular items, no lookahead
  4. MADRL-Cache      — Deep multi-agent RL caching (Zhong 2020)
  5. Submod-Greedy    — Submodular greedy with diversity guarantee
  6. SpaceCache+      — Coverage-prediction placement (Fang, INFOCOM 2024)
  7. Myopic-Optimal   — LP-optimal for current step only (H=0, K'=K)
  8. OLCP (H=5)       — our method: multi-horizon LP optimisation

Usage:
    python -m code.experiment --mode {main,ablation,scale,zipf,capacity,convergence,all}
"""

import sys
import os
import json
import argparse
import time as _time
from contextlib import contextmanager
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import (
    SAT_DIR, MAX_STEPS, REQUESTS_PER_STEP,
    CONTENT_SIZE_MB, GREEDY_REFRESH_INTERVAL, STEP_STRIDE,
    CONTENT_CATALOG_SIZE,
)
from . import config as cfg
from .common.data_loader import load_traces, get_nodes
from .common.topology import build_topology, get_cache_nodes, get_all_reachable_sats
from .common.metrics import generate_requests
from .common.popularity import PopularityTracker
from .olcp.solver import solve_olcp
from .olcp.router import route_olcp
from .baselines.no_cache import route_nocache
from .baselines.lce_lru import LCELRUManager
from .baselines.greedy_popular import greedy_placement, route_greedy
from .baselines.myopic import solve_myopic, route_myopic
from .baselines.drl_actor_critic import MaDRLManager, route_madrl
from .baselines.spacecache_plus import spacecache_placement, route_spacecache
from .baselines.submodular_greedy import submodular_greedy_placement, route_submodular


METHODS = ['nocache', 'lce_lru', 'greedy', 'madrl', 'submod',
           'spacecache', 'myopic', 'olcp']

METHOD_LABELS = {
    'nocache': 'No-Cache', 'lce_lru': 'LCE-LRU', 'greedy': 'Greedy-Pop',
    'madrl': 'MADRL-Cache', 'submod': 'Submod-Greedy',
    'spacecache': 'SpaceCache+', 'myopic': 'Myopic-Opt', 'olcp': 'OLCP (Ours)',
}

METHOD_COLORS = {
    'nocache': '#95a5a6', 'lce_lru': '#e74c3c', 'greedy': '#e67e22',
    'madrl': '#f39c12', 'submod': '#1abc9c',
    'spacecache': '#e91e63', 'myopic': '#3498db', 'olcp': '#2ecc71',
}


@contextmanager
def _runtime_overrides_from_args(args):
    previous = {
        'CACHE_CAPACITY': cfg.CACHE_CAPACITY,
        'LCE_LRU_CAPACITY': cfg.LCE_LRU_CAPACITY,
        'REQUESTS_PER_STEP': cfg.REQUESTS_PER_STEP,
        'ZIPF_ALPHA': cfg.ZIPF_ALPHA,
        'MIGRATION_BUDGET': cfg.MIGRATION_BUDGET,
    }
    if args.cache_capacity is not None:
        cfg.CACHE_CAPACITY = args.cache_capacity
        cfg.LCE_LRU_CAPACITY = args.cache_capacity
    if args.requests_per_step is not None:
        cfg.REQUESTS_PER_STEP = args.requests_per_step
    if args.zipf_alpha is not None:
        cfg.ZIPF_ALPHA = args.zipf_alpha
    if args.migration_budget is not None:
        cfg.MIGRATION_BUDGET = args.migration_budget
    try:
        yield
    finally:
        cfg.CACHE_CAPACITY = previous['CACHE_CAPACITY']
        cfg.LCE_LRU_CAPACITY = previous['LCE_LRU_CAPACITY']
        cfg.REQUESTS_PER_STEP = previous['REQUESTS_PER_STEP']
        cfg.ZIPF_ALPHA = previous['ZIPF_ALPHA']
        cfg.MIGRATION_BUDGET = previous['MIGRATION_BUDGET']


def _empty_results():
    return {m: {'delays': [], 'traffics': [], 'hits': 0, 'total': 0,
                'solve_time_ms': 0.0, 'per_step_hits': [], 'per_step_total': [],
                'cache_diversity': []} for m in METHODS}


def _count_unique_cached(placement):
    """Count unique content items across all nodes in a placement."""
    all_items = set()
    for contents in placement.values():
        all_items.update(contents)
    return len(all_items)


def _best_serving_node(G, requester, placement, content_id):
    """Return the cache node that would serve a hit under placement search."""
    if not G.has_node(requester):
        return None

    best_delay = float('inf')
    best_node = None
    for node, cached_items in placement.items():
        if content_id not in cached_items:
            continue
        if not G.has_node(node):
            continue
        try:
            path = nx.dijkstra_path(G, requester, node, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            if d < best_delay:
                best_delay = d
                best_node = node
        except nx.NetworkXNoPath:
            pass

    return best_node


def _build_future_snapshots(df_sat, df_uav, timestamps, start_idx, horizon):
    """Build topology snapshots for current + H future steps.
    
    Each snapshot includes both 'cache_nodes' (top-K for routing) and
    'all_reachable' (all nearby SATs for LP pre-positioning).
    """
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


def run_experiment(sat_dir=SAT_DIR, max_steps=MAX_STEPS):
    np.random.seed(42)
    import random; random.seed(42)
    df_sat, df_uav, timestamps = load_traces(sat_dir)
    if not timestamps:
        print("[ERROR] No timestamps.")
        return None
    # Subsample timestamps to get meaningful topology changes between steps
    timestamps = timestamps[::STEP_STRIDE][:max_steps]
    H = cfg.LOOKAHEAD_HORIZON
    print(f">>> OLCP experiment: {len(timestamps)} steps, H={H}")

    tracker = PopularityTracker()
    lce_mgr = LCELRUManager()
    madrl_mgr = MaDRLManager()
    results = _empty_results()

    # Cache states
    greedy_state = {}
    madrl_state = {}
    submod_state = {}
    spacecache_state = {}
    myopic_state = {}
    olcp_state = {}

    t0 = _time.time()

    for step in range(len(timestamps)):
        if step % 20 == 0:
            print(f"  Step {step}/{len(timestamps)} ({_time.time()-t0:.1f}s)")

        # Build future snapshots for OLCP
        snapshots = _build_future_snapshots(df_sat, df_uav, timestamps, step, H)
        if not snapshots:
            continue

        cur = snapshots[0]
        G = cur['G']
        type_map = cur['type_map']
        cache_nodes = cur['cache_nodes']
        if not cache_nodes:
            continue

        pop_scores = dict(tracker.scores)

        # ── Solve OLCP (uses all_reachable for pre-positioning) ──
        olcp_snapshots = []
        for s in snapshots:
            olcp_snapshots.append({
                'G': s['G'],
                'cache_nodes': s.get('all_reachable', s['cache_nodes']),
                'type_map': s['type_map'],
            })
        t_solve = _time.time()
        olcp_state, _ = solve_olcp(olcp_snapshots, olcp_state, pop_scores)
        results['olcp']['solve_time_ms'] += (_time.time() - t_solve) * 1000

        # ── Solve Myopic (H=0, uses all_reachable like OLCP for fair comparison) ──
        myopic_snap = {
            'G': cur['G'],
            'cache_nodes': cur.get('all_reachable', cur['cache_nodes']),
            'type_map': cur['type_map'],
        }
        t_solve = _time.time()
        myopic_state, _ = solve_myopic(myopic_snap, myopic_state, pop_scores)
        results['myopic']['solve_time_ms'] += (_time.time() - t_solve) * 1000

        all_reachable = cur.get('all_reachable', cache_nodes)

        # ── Greedy-Popular refresh ──
        if step % GREEDY_REFRESH_INTERVAL == 0:
            greedy_state = greedy_placement(cache_nodes, tracker)

        # ── MADRL-Cache (DQN per node, updates every step) ──
        madrl_state = madrl_mgr.decide_placement(all_reachable, pop_scores)

        # ── Submodular Greedy refresh ──
        if step % GREEDY_REFRESH_INTERVAL == 0:
            t_solve = _time.time()
            submod_state = submodular_greedy_placement(G, all_reachable, type_map, pop_scores)
            results['submod']['solve_time_ms'] += (_time.time() - t_solve) * 1000

        # ── SpaceCache+ refresh (coverage-prediction placement) ──
        if step % GREEDY_REFRESH_INTERVAL == 0:
            t_solve = _time.time()
            spacecache_state = spacecache_placement(G, all_reachable, type_map, pop_scores)
            results['spacecache']['solve_time_ms'] += (_time.time() - t_solve) * 1000

        # Track cache diversity
        results['greedy']['cache_diversity'].append(_count_unique_cached(greedy_state))
        results['madrl']['cache_diversity'].append(_count_unique_cached(madrl_state))
        results['submod']['cache_diversity'].append(_count_unique_cached(submod_state))
        results['spacecache']['cache_diversity'].append(_count_unique_cached(spacecache_state))
        results['myopic']['cache_diversity'].append(_count_unique_cached(myopic_state))
        results['olcp']['cache_diversity'].append(_count_unique_cached(olcp_state))

        # ── Generate requests ──
        requests = generate_requests(G, type_map)
        if not requests:
            continue
        tracker.decay_all()

        # Per-step hit counters
        step_hits = {m: 0 for m in METHODS}
        step_total = {m: 0 for m in METHODS}
        madrl_hit_counts = {c: 0 for c in all_reachable}

        for requester, cid in requests:
            tracker.record(cid)

            # No-Cache
            d, tr, hit = route_nocache(G, requester, type_map)
            if d is not None:
                results['nocache']['delays'].append(d)
                results['nocache']['traffics'].append(tr)
                results['nocache']['total'] += 1
                step_total['nocache'] += 1

            # LCE-LRU
            d, tr, hit = lce_mgr.route(G, requester, cache_nodes, cid, type_map)
            if d is not None:
                results['lce_lru']['delays'].append(d)
                results['lce_lru']['traffics'].append(tr)
                results['lce_lru']['hits'] += int(hit)
                results['lce_lru']['total'] += 1
                step_hits['lce_lru'] += int(hit)
                step_total['lce_lru'] += 1

            # Greedy-Popular
            d, tr, hit = route_greedy(G, requester, cache_nodes, greedy_state, cid, type_map)
            if d is not None:
                results['greedy']['delays'].append(d)
                results['greedy']['traffics'].append(tr)
                results['greedy']['hits'] += int(hit)
                results['greedy']['total'] += 1
                step_hits['greedy'] += int(hit)
                step_total['greedy'] += 1

            # MADRL-Cache
            madrl_serving_node = _best_serving_node(G, requester, madrl_state, cid)
            d, tr, hit = route_madrl(G, requester, madrl_state, cid, type_map)
            if d is not None:
                results['madrl']['delays'].append(d)
                results['madrl']['traffics'].append(tr)
                results['madrl']['hits'] += int(hit)
                results['madrl']['total'] += 1
                step_hits['madrl'] += int(hit)
                step_total['madrl'] += 1
                if hit and madrl_serving_node is not None:
                    madrl_hit_counts[madrl_serving_node] = madrl_hit_counts.get(madrl_serving_node, 0) + 1

            # Submodular Greedy
            d, tr, hit = route_submodular(G, requester, cache_nodes, submod_state, cid, type_map)
            if d is not None:
                results['submod']['delays'].append(d)
                results['submod']['traffics'].append(tr)
                results['submod']['hits'] += int(hit)
                results['submod']['total'] += 1
                step_hits['submod'] += int(hit)
                step_total['submod'] += 1

            # SpaceCache+
            d, tr, hit = route_spacecache(G, requester, spacecache_state, cid, type_map)
            if d is not None:
                results['spacecache']['delays'].append(d)
                results['spacecache']['traffics'].append(tr)
                results['spacecache']['hits'] += int(hit)
                results['spacecache']['total'] += 1
                step_hits['spacecache'] += int(hit)
                step_total['spacecache'] += 1

            # Myopic-Optimal
            d, tr, hit = route_myopic(G, requester, cache_nodes, myopic_state, cid, type_map)
            if d is not None:
                results['myopic']['delays'].append(d)
                results['myopic']['traffics'].append(tr)
                results['myopic']['hits'] += int(hit)
                results['myopic']['total'] += 1
                step_hits['myopic'] += int(hit)
                step_total['myopic'] += 1

            # OLCP
            d, tr, hit = route_olcp(G, requester, cache_nodes, olcp_state, cid, type_map)
            if d is not None:
                results['olcp']['delays'].append(d)
                results['olcp']['traffics'].append(tr)
                results['olcp']['hits'] += int(hit)
                results['olcp']['total'] += 1
                step_hits['olcp'] += int(hit)
                step_total['olcp'] += 1

        # Record per-step hit rates
        for m in METHODS:
            results[m]['per_step_hits'].append(step_hits[m])
            results[m]['per_step_total'].append(step_total[m])

        # MADRL end-of-step feedback: provide reward to each agent
        madrl_mgr.feedback(all_reachable, madrl_hit_counts, pop_scores)

    elapsed = _time.time() - t0
    print(f">>> Completed in {elapsed:.1f}s")
    return results


def compute_metrics(results):
    metrics = {}
    for m, d in results.items():
        n = d['total']
        if n == 0:
            metrics[m] = {'avg_delay_ms': 0, 'total_traffic_gb': 0, 'hit_rate': 0,
                          'backhaul_rate': 1.0, 'solve_time_ms': 0,
                          'avg_diversity': 0}
            continue
        div_list = d.get('cache_diversity', [])
        metrics[m] = {
            'avg_delay_ms': float(np.mean(d['delays'])) if d['delays'] else 0,
            'total_traffic_gb': float(np.sum(d['traffics'])) / 1024.0,
            'hit_rate': d['hits'] / n,
            'backhaul_rate': 1.0 - d['hits'] / n,
            'solve_time_ms': d['solve_time_ms'],
            'avg_diversity': float(np.mean(div_list)) if div_list else 0,
        }
    return metrics


def print_table(metrics):
    ORDER = METHODS
    print("\n" + "=" * 100)
    print("  OLCP Experiment Results")
    print("=" * 100)
    hdr = f"{'Method':<16} {'Delay(ms)':>12} {'Traffic(GB)':>12} {'Hit Rate':>10} {'Backhaul':>10} {'Diversity':>10} {'Solve(ms)':>10}"
    print(hdr)
    print("-" * 100)
    for m in ORDER:
        v = metrics[m]
        print(f"{METHOD_LABELS[m]:<16} {v['avg_delay_ms']:>12.1f} {v['total_traffic_gb']:>12.2f} "
              f"{v['hit_rate']:>9.1%} {v['backhaul_rate']:>9.1%} {v['avg_diversity']:>10.1f} {v['solve_time_ms']:>10.0f}")
    print("-" * 100)
    ours = metrics['olcp']
    nc = metrics['nocache']
    if nc['avg_delay_ms'] > 0:
        pct_d = (nc['avg_delay_ms'] - ours['avg_delay_ms']) / nc['avg_delay_ms'] * 100
        pct_t = (nc['total_traffic_gb'] - ours['total_traffic_gb']) / nc['total_traffic_gb'] * 100
        print(f"\n  OLCP vs No-Cache: delay ↓{pct_d:.1f}%, traffic ↓{pct_t:.1f}%")
    for m in ['lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic']:
        o = metrics[m]
        if o['avg_delay_ms'] > 0:
            d_imp = (o['avg_delay_ms'] - ours['avg_delay_ms']) / o['avg_delay_ms'] * 100
            h_imp = (ours['hit_rate'] - o['hit_rate']) * 100
            print(f"  OLCP vs {METHOD_LABELS[m]:<16}: delay ↓{d_imp:.1f}%, hit rate +{h_imp:.1f}pp")
    print("=" * 100)


def plot_results(metrics, output_dir=None):
    ORDER = METHODS
    LABELS = [METHOD_LABELS[m] for m in ORDER]
    COLORS = [METHOD_COLORS[m] for m in ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OLCP: Performance Comparison (100 LEO Satellites)',
                 fontsize=13, fontweight='bold')

    configs = [
        ('avg_delay_ms', 'Average Completion Delay', 'Delay (ms)', False),
        ('total_traffic_gb', 'Total Network Traffic', 'Traffic (GB)', False),
        ('hit_rate', 'Cache Hit Rate', 'Hit Rate (%)', True),
        ('backhaul_rate', 'Origin Backhaul Ratio', 'Backhaul (%)', True),
    ]
    for idx, (key, title, ylabel, is_pct) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        vals = [metrics[m][key] * 100 if is_pct else metrics[m][key] for m in ORDER]
        bars = ax.bar(range(len(LABELS)), vals, color=COLORS, width=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            fmt = f'{v:.1f}%' if is_pct else (f'{v:.2f}' if v < 100 else f'{v:.0f}')
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    fmt, ha='center', va='bottom', fontsize=7)
        best_idx = vals.index(max(vals)) if key == 'hit_rate' else vals.index(min(vals))
        bars[best_idx].set_edgecolor('#2ecc71')
        bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'experiment_results.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Figure saved: {out}")
    plt.close()


def plot_convergence(results, output_dir=None):
    """Plot per-step hit rate convergence over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    methods_to_plot = ['lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'olcp']
    window = 10  # moving average window

    for m in methods_to_plot:
        hits = results[m]['per_step_hits']
        totals = results[m]['per_step_total']
        if not hits:
            continue
        rates = [h / max(t, 1) for h, t in zip(hits, totals)]
        # Moving average
        if len(rates) >= window:
            smoothed = np.convolve(rates, np.ones(window) / window, mode='valid')
        else:
            smoothed = rates
        ax.plot(range(len(smoothed)), [r * 100 for r in smoothed],
                label=METHOD_LABELS[m], color=METHOD_COLORS[m], linewidth=1.5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cache Hit Rate (%)')
    ax.set_title('Hit Rate Convergence Over Time (Moving Avg, window=10)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'convergence.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Convergence figure saved: {out}")
    plt.close()


def plot_delay_cdf(results, output_dir=None):
    """Plot CDF of content delivery delay for all methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_to_plot = ['nocache', 'lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'olcp']

    for m in methods_to_plot:
        delays = sorted(results[m]['delays'])
        if not delays:
            continue
        cdf = np.arange(1, len(delays) + 1) / len(delays)
        ax.plot(delays, cdf, label=METHOD_LABELS[m], color=METHOD_COLORS[m], linewidth=1.5)

    ax.set_xlabel('Content Delivery Delay (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Content Delivery Delay')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'delay_cdf.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> CDF figure saved: {out}")
    plt.close()


def plot_diversity(results, output_dir=None):
    """Plot cache content diversity over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    methods_to_plot = ['greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'olcp']

    for m in methods_to_plot:
        div = results[m]['cache_diversity']
        if not div:
            continue
        ax.plot(range(len(div)), div, label=METHOD_LABELS[m],
                color=METHOD_COLORS[m], linewidth=1.5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Unique Cached Items')
    ax.set_title('Cache Content Diversity Over Time')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'cache_diversity.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Diversity figure saved: {out}")
    plt.close()


def plot_horizon_ablation(horizon_metrics, output_dir=None):
    """Plot ablation: hit rate & delay vs planning horizon H."""
    if not horizon_metrics:
        return
    Hs = sorted(horizon_metrics.keys())
    hit_rates = [horizon_metrics[h]['hit_rate'] * 100 for h in Hs]
    delays = [horizon_metrics[h]['avg_delay_ms'] for h in Hs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ablation: Effect of Planning Horizon H', fontsize=13, fontweight='bold')

    ax1.plot(Hs, hit_rates, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax1.set_xlabel('Planning Horizon H')
    ax1.set_ylabel('Cache Hit Rate (%)')
    ax1.set_title('Hit Rate vs Horizon')
    ax1.grid(alpha=0.3)
    for h, hr in zip(Hs, hit_rates):
        ax1.annotate(f'{hr:.1f}%', (h, hr), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

    ax2.plot(Hs, delays, 's-', color='#3498db', linewidth=2, markersize=8)
    ax2.set_xlabel('Planning Horizon H')
    ax2.set_ylabel('Average Delay (ms)')
    ax2.set_title('Delay vs Horizon')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'ablation_horizon.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Ablation figure saved: {out}")
    plt.close()


def save_metrics(metrics, filename='metrics.json', output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, filename)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f">>> Metrics saved: {out}")


def run_ablation_horizon(sat_dir=SAT_DIR, max_steps=MAX_STEPS, horizons=None):
    """Run OLCP with different horizon values for ablation study."""
    if horizons is None:
        horizons = [0, 1, 3, 5, 8]
    
    horizon_metrics = {}
    for h in horizons:
        print(f"\n>>> Ablation: H={h}")
        orig_h = cfg.LOOKAHEAD_HORIZON
        cfg.LOOKAHEAD_HORIZON = h
        results = run_experiment(sat_dir, max_steps)
        if results:
            m = compute_metrics(results)
            horizon_metrics[h] = m['olcp']
            print(f"    H={h}: hit={m['olcp']['hit_rate']:.3f}, delay={m['olcp']['avg_delay_ms']:.1f}")
        cfg.LOOKAHEAD_HORIZON = orig_h
    
    return horizon_metrics


def run_scale_experiment(sat_dirs, max_steps=MAX_STEPS):
    """Run experiment across different satellite counts."""
    scale_metrics = {}
    for sd in sat_dirs:
        label = os.path.basename(sd)
        print(f"\n>>> Scale experiment: {label}")
        results = run_experiment(sd, max_steps)
        if results:
            scale_metrics[label] = compute_metrics(results)
    return scale_metrics


def _scale_label_to_count(label):
    if label == 'sat_trace':
        return 25
    if label.startswith('sat_trace_'):
        try:
            return int(label.split('_')[-1])
        except ValueError:
            return float('inf')
    return float('inf')


def plot_scale_results(scale_metrics, output_dir=None, filename='scale_comparison.png'):
    """Plot comparison across satellite scales."""
    if not scale_metrics:
        return
    labels = sorted(scale_metrics.keys(), key=_scale_label_to_count)
    methods_to_plot = ['lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'olcp']
    colors = {m: METHOD_COLORS[m] for m in methods_to_plot}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Scalability: Performance vs Constellation Size', fontsize=13, fontweight='bold')
    
    x = range(len(labels))
    width = 0.15
    for i, m in enumerate(methods_to_plot):
        hits = [scale_metrics[l][m]['hit_rate'] * 100 for l in labels]
        delays = [scale_metrics[l][m]['avg_delay_ms'] for l in labels]
        offset = (i - 2) * width
        ax1.bar([xi + offset for xi in x], hits, width, label=METHOD_LABELS[m], color=colors[m])
        ax2.bar([xi + offset for xi in x], delays, width, color=colors[m])
    
    xlabels = [f'{_scale_label_to_count(label)} SATs' for label in labels]
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels)
    ax1.set_ylabel('Cache Hit Rate (%)'); ax1.set_title('Hit Rate')
    ax1.legend(fontsize=7); ax1.grid(axis='y', alpha=0.3)
    
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels)
    ax2.set_ylabel('Average Delay (ms)'); ax2.set_title('Completion Delay')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Scale figure saved: {out}")
    plt.close()


# ── New sensitivity experiments ──

def run_zipf_sensitivity(sat_dir=SAT_DIR, max_steps=MAX_STEPS, alphas=None):
    """Vary Zipf skewness α and measure hit rate for all methods."""
    if alphas is None:
        alphas = [1.1, 1.3, 1.5, 1.8, 2.2]
    
    zipf_metrics = {}
    for alpha in alphas:
        print(f"\n>>> Zipf sensitivity: α={alpha}")
        orig_alpha = cfg.ZIPF_ALPHA
        cfg.ZIPF_ALPHA = alpha
        results = run_experiment(sat_dir, max_steps)
        if results:
            zipf_metrics[alpha] = compute_metrics(results)
        cfg.ZIPF_ALPHA = orig_alpha
    return zipf_metrics


def plot_zipf_sensitivity(zipf_metrics, output_dir=None):
    """Plot hit rate vs Zipf skewness for all methods."""
    if not zipf_metrics:
        return
    alphas = sorted(zipf_metrics.keys())
    methods_to_plot = ['lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'olcp']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Sensitivity: Impact of Zipf Skewness α', fontsize=13, fontweight='bold')

    for m in methods_to_plot:
        hits = [zipf_metrics[a][m]['hit_rate'] * 100 for a in alphas]
        delays = [zipf_metrics[a][m]['avg_delay_ms'] for a in alphas]
        ax1.plot(alphas, hits, 'o-', label=METHOD_LABELS[m], color=METHOD_COLORS[m], linewidth=1.5, markersize=6)
        ax2.plot(alphas, delays, 's-', color=METHOD_COLORS[m], linewidth=1.5, markersize=6)

    ax1.set_xlabel('Zipf Skewness α'); ax1.set_ylabel('Cache Hit Rate (%)')
    ax1.set_title('Hit Rate vs α'); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    ax2.set_xlabel('Zipf Skewness α'); ax2.set_ylabel('Average Delay (ms)')
    ax2.set_title('Delay vs α'); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'zipf_sensitivity.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Zipf sensitivity figure saved: {out}")
    plt.close()


def run_capacity_sensitivity(sat_dir=SAT_DIR, max_steps=MAX_STEPS, caps=None):
    """Vary cache capacity C_cap and measure hit rate for all methods."""
    if caps is None:
        caps = [5, 10, 20, 30, 40]
    
    cap_metrics = {}
    for cap in caps:
        print(f"\n>>> Capacity sensitivity: C_cap={cap}")
        orig_cap = cfg.CACHE_CAPACITY
        orig_lce_cap = cfg.LCE_LRU_CAPACITY
        cfg.CACHE_CAPACITY = cap
        cfg.LCE_LRU_CAPACITY = cap
        results = run_experiment(sat_dir, max_steps)
        if results:
            cap_metrics[cap] = compute_metrics(results)
        cfg.CACHE_CAPACITY = orig_cap
        cfg.LCE_LRU_CAPACITY = orig_lce_cap
    return cap_metrics


def plot_capacity_sensitivity(cap_metrics, output_dir=None):
    """Plot hit rate vs cache capacity for all methods."""
    if not cap_metrics:
        return
    caps = sorted(cap_metrics.keys())
    methods_to_plot = ['lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'olcp']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Sensitivity: Impact of Cache Capacity C_cap', fontsize=13, fontweight='bold')

    for m in methods_to_plot:
        hits = [cap_metrics[c][m]['hit_rate'] * 100 for c in caps]
        delays = [cap_metrics[c][m]['avg_delay_ms'] for c in caps]
        ax1.plot(caps, hits, 'o-', label=METHOD_LABELS[m], color=METHOD_COLORS[m], linewidth=1.5, markersize=6)
        ax2.plot(caps, delays, 's-', color=METHOD_COLORS[m], linewidth=1.5, markersize=6)

    ax1.set_xlabel('Cache Capacity (items)'); ax1.set_ylabel('Cache Hit Rate (%)')
    ax1.set_title('Hit Rate vs C_cap'); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    ax2.set_xlabel('Cache Capacity (items)'); ax2.set_ylabel('Average Delay (ms)')
    ax2.set_title('Delay vs C_cap'); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(output_dir, 'capacity_sensitivity.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f">>> Capacity sensitivity figure saved: {out}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='main',
                        choices=['main', 'ablation', 'scale', 'zipf', 'capacity', 'all'])
    parser.add_argument('--sat-dir', default=SAT_DIR)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--requests-per-step', type=int, default=None)
    parser.add_argument('--cache-capacity', type=int, default=None)
    parser.add_argument('--zipf-alpha', type=float, default=None)
    parser.add_argument('--migration-budget', type=int, default=None)
    parser.add_argument('--output-suffix', default='')
    args = parser.parse_args()

    # ── Output directories ──
    # Persist figures and metrics at the project root so they integrate with
    # the existing S3 figures/ and results/ workflow.
    _code_dir = os.path.dirname(os.path.abspath(__file__))
    _algorithms_dir = os.path.dirname(_code_dir)
    _project_root = os.path.dirname(_algorithms_dir)
    figures_dir = os.path.join(_project_root, 'figures')
    results_dir = os.path.join(_project_root, 'results')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ''

    with _runtime_overrides_from_args(args):
        if args.mode in ('main', 'all'):
            results = run_experiment(args.sat_dir, args.max_steps)
            if results:
                metrics = compute_metrics(results)
                print_table(metrics)
                plot_results(metrics, figures_dir)
                plot_convergence(results, figures_dir)
                plot_delay_cdf(results, figures_dir)
                plot_diversity(results, figures_dir)
                save_metrics(metrics, f'metrics{suffix}.json', results_dir)

        if args.mode in ('ablation', 'all'):
            hm = run_ablation_horizon(args.sat_dir, args.max_steps)
            plot_horizon_ablation(hm, figures_dir)
            with open(os.path.join(results_dir, f'ablation_metrics{suffix}.json'), 'w') as f:
                json.dump({str(k): v for k, v in hm.items()}, f, indent=2)
            print(">>> Ablation metrics saved.")

        if args.mode in ('scale', 'all'):
            traces_dir = os.path.join(_project_root, 'traces')
            sat_dirs = []
            for d in ['sat_trace', 'sat_trace_50', 'sat_trace_100', 'sat_trace_150']:
                p = os.path.join(traces_dir, d)
                if os.path.isdir(p):
                    sat_dirs.append(p)
            if sat_dirs:
                sm = run_scale_experiment(sat_dirs, args.max_steps)
                plot_scale_results(sm, figures_dir, filename=f'scale_comparison{suffix}.png')
                with open(os.path.join(results_dir, f'scale_metrics{suffix}.json'), 'w') as f:
                    json.dump(sm, f, indent=2)
                print(">>> Scale metrics saved.")

        if args.mode in ('zipf', 'all'):
            zm = run_zipf_sensitivity(args.sat_dir, args.max_steps)
            plot_zipf_sensitivity(zm, figures_dir)
            with open(os.path.join(results_dir, f'zipf_metrics{suffix}.json'), 'w') as f:
                json.dump({str(k): v for k, v in zm.items()}, f, indent=2)
            print(">>> Zipf sensitivity metrics saved.")

        if args.mode in ('capacity', 'all'):
            cm = run_capacity_sensitivity(args.sat_dir, args.max_steps)
            plot_capacity_sensitivity(cm, figures_dir)
            with open(os.path.join(results_dir, f'capacity_metrics{suffix}.json'), 'w') as f:
                json.dump({str(k): v for k, v in cm.items()}, f, indent=2)
            print(">>> Capacity sensitivity metrics saved.")

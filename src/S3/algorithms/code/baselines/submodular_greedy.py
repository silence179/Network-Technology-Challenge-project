"""
Baseline: Submodular Greedy — Greedy submodular maximization for cooperative caching.

Maximizes a monotone submodular coverage objective:
  Repeatedly select the (node, content) pair with the largest marginal gain
  until all cache budgets are exhausted.

Achieves (1-1/e) ≈ 63.2% approximation guarantee for unconstrained submodular.
With partition matroid (per-node capacity), guarantee still holds.

Reference: Nemhauser, Wolsey & Fisher, "An Analysis of Approximations for
Maximizing Submodular Set Functions", Math. Programming 1978
"""

import numpy as np
import networkx as nx

from ..config import (
    CONTENT_CATALOG_SIZE,
    CONTENT_SIZE_BITS, CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS,
    CONTENT_SIZE_MB,
)
from .. import config as cfg

F = CONTENT_CATALOG_SIZE


def _compute_delay_savings(G, cache_nodes, type_map):
    """Compute per-node delay savings for marginal gain computation."""
    uav_nodes = [n for n, t in type_map.items() if t == 'UAV' and G.has_node(n)]
    if not uav_nodes:
        return {}

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    miss_delays = []
    for u in uav_nodes:
        best = float('inf')
        for gs in gs_nodes:
            try:
                d = nx.dijkstra_path_length(G, u, gs, weight='eff_delay')
                best = min(best, d)
            except nx.NetworkXNoPath:
                pass
        if best < float('inf'):
            miss_delays.append(2 * best + transfer_origin)
    d_miss = np.mean(miss_delays) if miss_delays else 10000.0

    delta = {}
    for c in cache_nodes:
        if not G.has_node(c):
            continue
        hit_d = []
        for u in uav_nodes:
            try:
                d = nx.dijkstra_path_length(G, u, c, weight='eff_delay')
                hit_d.append(d + transfer_cache)
            except nx.NetworkXNoPath:
                pass
        delta[c] = max(d_miss - np.mean(hit_d), 0.0) if hit_d else 0.0

    return delta


def submodular_greedy_placement(G, cache_nodes, type_map, popularity_scores):
    """Greedy submodular placement with diversity.

    At each step, pick the (node, content) pair with the highest marginal gain:
        gain(c, f) = λ_f · δ_c · (1 if f not yet placed anywhere)

    The "1 if f not yet placed" term is what makes this submodular — once f is
    already cached somewhere, placing another copy has zero marginal gain.
    """
    lam = np.zeros(F)
    for fid, sc in popularity_scores.items():
        if 0 <= fid < F:
            lam[fid] = sc
    if lam.sum() == 0:
        lam[:] = 1.0

    delta = _compute_delay_savings(G, cache_nodes, type_map)

    C_CAP = cfg.CACHE_CAPACITY
    placement = {c: set() for c in cache_nodes}
    placed_globally = set()  # contents placed anywhere
    budget = {c: C_CAP for c in cache_nodes}

    # Greedy: iterate up to total budget
    total_budget = sum(budget.values())
    for _ in range(total_budget):
        best_gain = -1
        best_c, best_f = None, None

        for c in cache_nodes:
            if budget[c] <= 0:
                continue
            dc = delta.get(c, 0.0)
            if dc == 0:
                continue
            for f in range(F):
                if f in placed_globally:
                    continue  # no marginal gain for duplicates
                if f in placement[c]:
                    continue
                gain = lam[f] * dc
                if gain > best_gain:
                    best_gain = gain
                    best_c, best_f = c, f

        if best_c is None or best_gain <= 0:
            # Fill remaining with best available (allow duplicates as fallback)
            for c in cache_nodes:
                if budget[c] <= 0:
                    continue
                remaining = [(lam[f], f) for f in range(F) if f not in placement[c]]
                remaining.sort(reverse=True)
                for _, f in remaining[:budget[c]]:
                    placement[c].add(f)
                budget[c] = 0
            break

        placement[best_c].add(best_f)
        placed_globally.add(best_f)
        budget[best_c] -= 1

    return placement


def route_submodular(G, requester, cache_nodes, placement, content_id, type_map):
    """Route using submodular greedy placement."""
    if not G.has_node(requester):
        return None, None, False

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    best_delay, best_path = float('inf'), None
    for c, cached in placement.items():
        if content_id not in cached:
            continue
        if not G.has_node(c):
            continue
        try:
            path = nx.dijkstra_path(G, requester, c, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = d + transfer_cache
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass

    if best_path is not None:
        traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
        return best_delay, traffic, True

    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    best_delay, best_path = float('inf'), None
    for gs in gs_nodes:
        try:
            path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = 2 * d + transfer_origin
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass
    if best_path is None:
        return None, None, False
    traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
    return best_delay, traffic, False

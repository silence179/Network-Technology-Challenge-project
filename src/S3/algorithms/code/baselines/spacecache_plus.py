"""
Faithful implementation of SpaceCache+: Coverage-Prediction Content Placement
for LEO Mega-Constellations.

Reference:
  Fang, H., Wang, F., Liu, J.,
  "SpaceCache+: Towards Pervasive Content Delivery via Low-Earth Orbit
   Mega-Constellations", Proc. IEEE INFOCOM, pp. 1-10, 2024.

Algorithm:
  1. Coverage Prediction — for each satellite, estimate the user coverage:
     - Number of reachable UAV/ground nodes
     - Link quality (inverse delay) to those users
     - Coverage score = Σ_u (1 / delay(sat, u)) for reachable users u
  2. Coverage-Weighted Demand — per satellite content demand:
     - demand(c, f) = coverage_score(c) × popularity(f)
  3. Diversity-Aware Greedy Placement:
     - Sort satellites by coverage score (high → low)
     - For each satellite, select top-C_cap content by demand score
     - After placing content f on satellite c, reduce f's demand score
       on overlapping-coverage satellites by a redundancy penalty
     - This encourages content diversity across satellites with
       overlapping coverage areas
  4. Coverage Graph — build per-step coverage overlap graph:
     - Two satellites overlap if they share ≥ 1 reachable user
     - Redundancy penalty only applies between overlapping satellites
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


# ─────────────────── Coverage Analysis ───────────────────

def _compute_coverage(G, cache_nodes, type_map):
    """Compute per-satellite coverage scores and user reachability.

    Returns
    -------
    coverage_scores : dict {node: float}
        Coverage score = Σ_u 1/delay(c, u) for all reachable UAV users u.
    user_sets : dict {node: set}
        The set of UAV users reachable from each cache satellite.
    """
    uav_nodes = [n for n, t in type_map.items() if t == 'UAV' and G.has_node(n)]
    if not uav_nodes:
        return {}, {}

    coverage_scores = {}
    user_sets = {}

    for c in cache_nodes:
        if not G.has_node(c):
            coverage_scores[c] = 0.0
            user_sets[c] = set()
            continue

        score = 0.0
        users = set()
        for u in uav_nodes:
            try:
                d = nx.dijkstra_path_length(G, c, u, weight='eff_delay')
                if d > 0:
                    score += 1.0 / d     # inverse-delay coverage metric
                    users.add(u)
            except nx.NetworkXNoPath:
                pass

        coverage_scores[c] = score
        user_sets[c] = users

    return coverage_scores, user_sets


def _build_overlap_graph(cache_nodes, user_sets):
    """Build coverage-overlap graph: edge between c1, c2 if they share users.

    Returns adjacency dict {node: set(overlapping_neighbors)}.
    """
    overlap = {c: set() for c in cache_nodes}
    nodes = list(cache_nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            c1, c2 = nodes[i], nodes[j]
            if user_sets.get(c1, set()) & user_sets.get(c2, set()):
                overlap[c1].add(c2)
                overlap[c2].add(c1)
    return overlap


# ─────────────────── Content Placement ───────────────────

def spacecache_placement(G, cache_nodes, type_map, popularity_scores,
                         redundancy_penalty=0.15):
    """SpaceCache+ coverage-prediction content placement.

    Parameters
    ----------
    G : nx.Graph
    cache_nodes : set/list
    type_map : dict
    popularity_scores : dict {content_id: float}
    redundancy_penalty : float
        Discount factor for placing same content on overlapping-coverage sats.
        0 = no penalty (allow full redundancy)
        1 = full penalty (never duplicate on overlapping sats)

    Returns
    -------
    placement : dict {node: set(content_ids)}
    """
    C_CAP = cfg.CACHE_CAPACITY

    # ── Step 1: Coverage prediction ──
    coverage_scores, user_sets = _compute_coverage(G, cache_nodes, type_map)

    # ── Step 2: Coverage overlap graph ──
    overlap = _build_overlap_graph(cache_nodes, user_sets)

    # ── Step 3: Content popularity vector ──
    pop = np.zeros(F)
    for fid, sc in popularity_scores.items():
        if 0 <= fid < F:
            pop[fid] = sc
    if pop.sum() == 0:
        pop[:] = 1.0

    # ── Step 4: Diversity-aware greedy placement ──
    # Sort satellites by coverage score (highest first → most influential)
    sorted_sats = sorted(cache_nodes, key=lambda c: coverage_scores.get(c, 0),
                         reverse=True)

    placement = {}
    # Track how many times each content has been placed on overlapping sats
    placed_on_overlap = np.zeros(F)    # accumulated redundancy penalty per content

    for c in sorted_sats:
        cov = coverage_scores.get(c, 0.0)
        if cov == 0:
            # Satellite has no user coverage — fill with top popular
            top = np.argsort(pop)[::-1][:C_CAP]
            placement[c] = set(int(x) for x in top)
            continue

        # Demand score = coverage × popularity × (1 - redundancy_penalty × overlap_count)
        demand = np.zeros(F)
        for f in range(F):
            penalty = 1.0 - redundancy_penalty * placed_on_overlap[f]
            penalty = max(penalty, 0.1)    # floor to avoid zeroing out
            demand[f] = cov * pop[f] * penalty

        # Select top-C_cap by demand
        selected = np.argsort(demand)[::-1][:C_CAP]
        placement[c] = set(int(x) for x in selected)

        # Update redundancy penalties for overlapping satellites
        for f in selected:
            # Increment for all content placed at this satellite
            for neighbor in overlap.get(c, set()):
                if neighbor not in placement:    # only penalize unplaced sats
                    placed_on_overlap[int(f)] += 1.0

    return placement


# ─────────────────── Routing ───────────────────

def route_spacecache(G, requester, placement, content_id, type_map):
    """Route request using SpaceCache+ placement (search all placed nodes)."""
    if not G.has_node(requester):
        return None, None, False

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    # Check all placement nodes
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

    # Origin fallback
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

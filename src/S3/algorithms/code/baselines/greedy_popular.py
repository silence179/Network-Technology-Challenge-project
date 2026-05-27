"""
Baseline: Greedy-Popular — periodically fill caches with most popular content.
No lookahead; uses only current popularity scores.
"""

import networkx as nx

from .. import config as cfg
from ..config import (
    CONTENT_SIZE_BITS, CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS,
    CONTENT_SIZE_MB, GREEDY_REFRESH_INTERVAL,
)


def greedy_placement(cache_nodes, popularity_tracker):
    """Fill each cache node with the globally top-C_cap popular items."""
    top_items = popularity_tracker.top_k(cfg.CACHE_CAPACITY)
    placement = {}
    for c in cache_nodes:
        placement[c] = set(top_items)
    return placement


def route_greedy(G, requester, cache_nodes, placement, content_id, type_map):
    """Route using greedy-popular placement (same search as OLCP router)."""
    if not G.has_node(requester):
        return None, None, False

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    # Check caches
    best_delay, best_path = float('inf'), None
    for c in cache_nodes:
        if content_id not in placement.get(c, set()):
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

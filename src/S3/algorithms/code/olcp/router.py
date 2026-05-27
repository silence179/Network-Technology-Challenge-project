"""
OLCP Router: routes requests using the OLCP cache placement.

Cache search order:
  1. Check all cache nodes (sorted by delay)
  2. Origin fallback via Dijkstra
"""

import networkx as nx

from ..config import CONTENT_SIZE_BITS, CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS, CONTENT_SIZE_MB


def route_olcp(G, requester, cache_nodes, placement, content_id, type_map):
    """Route a request using OLCP placement.

    Checks ALL nodes in placement (not just top-K), since OLCP may
    pre-position content at nearby satellites beyond the closest K.

    Returns (delay_ms, traffic_mb, is_hit) or (None, None, False) on failure.
    """
    if not G.has_node(requester):
        return None, None, False

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    # Stage 1: search ALL placed nodes for content (not just top-K)
    best_delay = float('inf')
    best_path = None
    for c, cached in placement.items():
        if content_id not in cached:
            continue
        if not G.has_node(c):
            continue
        try:
            path = nx.dijkstra_path(G, requester, c, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = d + transfer_cache  # one-way
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass

    if best_path is not None:
        traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
        return best_delay, traffic, True

    # Stage 2: origin fallback
    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    best_delay = float('inf')
    best_path = None
    for gs in gs_nodes:
        try:
            path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = 2 * d + transfer_origin  # round-trip
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass

    if best_path is not None:
        traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
        return best_delay, traffic, False

    return None, None, False

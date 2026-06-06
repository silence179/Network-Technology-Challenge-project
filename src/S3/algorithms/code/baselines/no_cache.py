"""
Baseline: No-Cache — pure Dijkstra routing to origin.
"""

import networkx as nx
from ..config import CONTENT_SIZE_BITS, GS_SERVE_BW_MBPS, CONTENT_SIZE_MB


def route_nocache(G, requester, type_map):
    """Route via shortest delay path to ground station, no caching."""
    if not G.has_node(requester):
        return None, None, False
    transfer = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0
    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    best_delay, best_path = float('inf'), None
    for gs in gs_nodes:
        try:
            path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = 2 * d + transfer
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass
    if best_path is None:
        return None, None, False
    traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
    return best_delay, traffic, False

"""
Baseline: LCE-LRU — Leave Copy Everywhere with LRU eviction (classic NDN).
"""

import networkx as nx
from collections import OrderedDict

from .. import config as cfg
from ..config import (
    CONTENT_SIZE_BITS, CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS,
    CONTENT_SIZE_MB,
)


class LCEStore:
    """Per-node LRU cache store."""

    def __init__(self, capacity=None):
        self.cap = capacity if capacity is not None else cfg.CACHE_CAPACITY
        self.store = OrderedDict()

    def has(self, cid):
        if cid in self.store:
            self.store.move_to_end(cid)
            return True
        return False

    def put(self, cid):
        if cid in self.store:
            self.store.move_to_end(cid)
            return
        if len(self.store) >= self.cap:
            self.store.popitem(last=False)
        self.store[cid] = True

    def contents(self):
        return set(self.store.keys())


class LCELRUManager:
    """Manages LCE-LRU stores across all SAT nodes."""

    def __init__(self):
        self._stores = {}

    def _get(self, node):
        if node not in self._stores:
            self._stores[node] = LCEStore()
        return self._stores[node]

    def route(self, G, requester, cache_nodes, content_id, type_map):
        """Route with LCE-LRU: check path to origin, cache everywhere on return."""
        if not G.has_node(requester):
            return None, None, False

        transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
        transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

        # Find path to GS (check caches along the way)
        gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
        best_path, best_delay = None, float('inf')
        for gs in gs_nodes:
            try:
                path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
                d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
                if d < best_delay:
                    best_delay = d
                    best_path = path
            except nx.NetworkXNoPath:
                pass
        if best_path is None:
            return None, None, False

        # Check for cache hit along the path
        for idx, node in enumerate(best_path):
            if type_map.get(node) == 'SAT' and self._get(node).has(content_id):
                # Cache hit at this node
                sub_path = best_path[:idx + 1]
                d = sum(G[sub_path[i]][sub_path[i + 1]]['eff_delay'] for i in range(len(sub_path) - 1))
                traffic = CONTENT_SIZE_MB * (len(sub_path) - 1)
                return d + transfer_cache, traffic, True

        # Cache miss: fetch from origin
        total = 2 * best_delay + transfer_origin
        traffic = CONTENT_SIZE_MB * (len(best_path) - 1)

        # Leave Copy Everywhere: cache at every SAT on the return path
        for node in best_path:
            if type_map.get(node) == 'SAT':
                self._get(node).put(content_id)

        return total, traffic, False

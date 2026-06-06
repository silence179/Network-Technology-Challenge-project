"""
Request generation and path metrics.
"""

import random
import numpy as np
import networkx as nx

from ..config import (
    REQUESTS_PER_STEP, CONTENT_CATALOG_SIZE, CONTENT_SIZE_MB,
    CONTENT_SIZE_BITS
)
from .. import config as cfg


def generate_requests(G, type_map, n_req=None):
    """Generate Zipf-distributed content requests from UAV nodes."""
    uav_nodes = [nid for nid, t in type_map.items() if t == 'UAV' and G.has_node(nid)]
    if not uav_nodes:
        return []
    request_count = n_req if n_req is not None else getattr(cfg, 'REQUESTS_PER_STEP', REQUESTS_PER_STEP)
    reqs = []
    alpha = getattr(cfg, 'ZIPF_ALPHA', 1.5)
    for _ in range(request_count):
        requester = random.choice(uav_nodes)
        content_id = int(np.random.zipf(alpha)) % CONTENT_CATALOG_SIZE
        reqs.append((requester, content_id))
    return reqs


def path_completion_time(G, path, rtt=True, serve_bw=None):
    """Compute content delivery completion time along a path."""
    if len(path) < 2:
        return 0.0
    prop = sum(G[path[i]][path[i+1]]['eff_delay'] for i in range(len(path) - 1))
    bottleneck = min(G[path[i]][path[i+1]].get('bw', 1) for i in range(len(path) - 1))
    bw = max(bottleneck, serve_bw) if serve_bw else bottleneck
    transfer = (CONTENT_SIZE_BITS / (bw * 1e6)) * 1000.0
    return (2 * prop if rtt else prop) + transfer


def path_traffic_mb(path):
    """Traffic in MB based on hop count (each hop transmits full content)."""
    hops = len(path) - 1
    return CONTENT_SIZE_MB * hops

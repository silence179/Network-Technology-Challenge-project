"""
Topology builder: construct NetworkX graph from node positions.
"""

import networkx as nx
from scipy.spatial import cKDTree

from ..config import MAX_LINK_RANGE, MIN_ELEVATION, CACHE_SAT_COUNT
from .network_utils import (
    propagation_delay_ms, link_bandwidth_mbps, elevation_deg,
    packet_loss_rate, effective_delay_ms,
)


def build_topology(nodes_df):
    """Build a weighted undirected graph from node DataFrame at a single timestep."""
    G = nx.Graph()
    if nodes_df.empty:
        return G, {}, {}

    coords = nodes_df[['ecef_x', 'ecef_y', 'ecef_z']].values
    ids = nodes_df['node_id'].values
    types = nodes_df['type'].values

    coord_map = {ids[i]: coords[i] for i in range(len(ids))}
    type_map = {ids[i]: types[i] for i in range(len(ids))}

    for nid in ids:
        G.add_node(nid, node_type=type_map[nid])

    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=20, distance_upper_bound=MAX_LINK_RANGE)

    processed = set()
    for i in range(len(ids)):
        for j_pos, j in enumerate(indices[i]):
            if dists[i][j_pos] == float('inf') or i == j:
                continue
            n1, n2 = ids[i], ids[j]
            if (n1, n2) in processed or (n2, n1) in processed:
                continue
            ta, tb = types[i], types[j]
            if (ta == 'SAT') != (tb == 'SAT'):
                sat_idx, gnd_idx = (i, j) if ta == 'SAT' else (j, i)
                if elevation_deg(coords[gnd_idx], coords[sat_idx]) < MIN_ELEVATION:
                    continue
            bw = link_bandwidth_mbps(ta, tb)
            if bw == 0:
                continue
            dist_m = dists[i][j_pos]
            delay = propagation_delay_ms(dist_m)
            loss = packet_loss_rate(ta, tb)
            eff_delay = effective_delay_ms(delay, loss)
            G.add_edge(n1, n2, delay=delay, bw=bw, dist_m=dist_m,
                       loss=loss, eff_delay=eff_delay)
            processed.add((n1, n2))

    return G, coord_map, type_map


def get_cache_nodes(G, type_map, n=CACHE_SAT_COUNT):
    """Select the n best SAT cache nodes based on minimum delay to UAVs."""
    uav_nodes = [nid for nid, t in type_map.items() if t == 'UAV' and G.has_node(nid)]
    sat_nodes = [nid for nid, t in type_map.items() if t == 'SAT' and G.has_node(nid)]
    if not sat_nodes or not uav_nodes:
        return set()

    candidates = set()
    for uav in uav_nodes:
        for nb in G.neighbors(uav):
            if type_map.get(nb) == 'SAT':
                candidates.add(nb)
    if not candidates:
        return set()

    scored = []
    for sat in candidates:
        d = min(
            (G[uav][sat]['eff_delay'] for uav in uav_nodes if G.has_edge(uav, sat)),
            default=float('inf')
        )
        scored.append((d, sat))
    scored.sort()
    return set(s for _, s in scored[:n])


def get_all_reachable_sats(G, type_map, max_n=None):
    """Return an adaptive expanded cache set of SAT nodes closest to UAVs.

    This is a superset of top-K cache nodes, used for LP pre-positioning.
    The cap grows with the reachable candidate pool so larger constellations
    still expose additional caching choices instead of flattening the scale study.
    """
    uav_nodes = [nid for nid, t in type_map.items() if t == 'UAV' and G.has_node(nid)]
    candidates = set()
    for nid in uav_nodes:
        for nb in G.neighbors(nid):
            if type_map.get(nb) == 'SAT':
                candidates.add(nb)
    if not candidates:
        return set()
    if max_n is None:
        sat_count = sum(1 for nid, node_type in type_map.items() if node_type == 'SAT' and G.has_node(nid))
        dynamic_limit = max(CACHE_SAT_COUNT * 2, (sat_count // 10) + CACHE_SAT_COUNT + 1)
        max_n = min(14, dynamic_limit)
    # Sort by average delay to UAVs, keep top max_n
    scored = []
    for sat in candidates:
        d = min(
            (G[uav][sat]['eff_delay'] for uav in uav_nodes if G.has_edge(uav, sat)),
            default=float('inf')
        )
        scored.append((d, sat))
    scored.sort()
    return set(s for _, s in scored[:max_n])

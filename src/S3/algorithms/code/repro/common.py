from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from ..config import MAX_STEPS, SAT_DIR, STEP_STRIDE
from ..common.data_loader import get_nodes, load_traces
from ..common.topology import build_topology, get_all_reachable_sats, get_cache_nodes


@dataclass
class StepSnapshot:
    t_ms: int
    G: nx.Graph
    type_map: dict[str, str]
    coord_map: dict[str, tuple[float, float, float]]
    cache_nodes: set[str]
    all_reachable: set[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_sampled_traces(sat_dir: str = SAT_DIR, max_steps: int = MAX_STEPS):
    df_sat, df_uav, timestamps = load_traces(sat_dir)
    sampled = timestamps[::STEP_STRIDE][:max_steps]
    return df_sat, df_uav, sampled


def build_snapshot(df_sat, df_uav, t_ms: int) -> StepSnapshot | None:
    nodes_df = get_nodes(df_sat, df_uav, int(t_ms))
    if nodes_df.empty:
        return None

    G, coord_map, type_map = build_topology(nodes_df)
    if len(G.nodes) < 2:
        return None

    cache_nodes = get_cache_nodes(G, type_map)
    all_reachable = get_all_reachable_sats(G, type_map)
    if not all_reachable:
        return None

    return StepSnapshot(
        t_ms=int(t_ms),
        G=G,
        type_map=type_map,
        coord_map={k: tuple(float(v) for v in values) for k, values in coord_map.items()},
        cache_nodes=set(cache_nodes),
        all_reachable=set(all_reachable),
    )


def best_serving_node(G: nx.Graph, requester: str, placement: dict[str, set[int]], content_id: int):
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
            delay = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            if delay < best_delay:
                best_delay = delay
                best_node = node
        except nx.NetworkXNoPath:
            pass
    return best_node


def count_unique_cached(placement: dict[str, set[int]]) -> int:
    all_items: set[int] = set()
    for contents in placement.values():
        all_items.update(contents)
    return len(all_items)


def compute_basic_metrics(
    delays: list[float],
    traffics: list[float],
    hits: int,
    total: int,
    diversity: list[int],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        'avg_delay_ms': float(np.mean(delays)) if delays else 0.0,
        'total_traffic_gb': float(np.sum(traffics)) / 1024.0 if traffics else 0.0,
        'hit_rate': hits / total if total else 0.0,
        'backhaul_rate': 1.0 - (hits / total if total else 0.0),
        'served_requests': total,
        'avg_diversity': float(np.mean(diversity)) if diversity else 0.0,
    }
    if extra:
        metrics.update(extra)
    return metrics
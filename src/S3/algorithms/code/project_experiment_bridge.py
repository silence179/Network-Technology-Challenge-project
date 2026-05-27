from __future__ import annotations

import time
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

from . import config as cfg
from .baselines.drl_actor_critic import MaDRLManager
from .baselines.greedy_popular import greedy_placement
from .baselines.lce_lru import LCELRUManager
from .baselines.myopic import solve_myopic
from .baselines.spacecache_plus import spacecache_placement
from .baselines.submodular_greedy import submodular_greedy_placement
from .common.popularity import PopularityTracker
from .common.topology import get_all_reachable_sats, get_cache_nodes
from .olcp.solver import solve_olcp


SUPPORTED_CONTENT_METHODS = (
    "nocache",
    "lce_lru",
    "greedy",
    "madrl",
    "submod",
    "spacecache",
    "myopic",
    "otcp",
)

METHOD_LABELS = {
    "nocache": "No-Cache",
    "lce_lru": "LCE-LRU",
    "greedy": "Greedy-Pop",
    "madrl": "MADRL-Cache",
    "submod": "Submod-Greedy",
    "spacecache": "SpaceCache+",
    "myopic": "Myopic-Opt",
    "otcp": "OTCP/OLCP",
}

METHOD_COLORS = {
    "nocache": "#95a5a6",
    "lce_lru": "#e74c3c",
    "greedy": "#e67e22",
    "madrl": "#f39c12",
    "submod": "#1abc9c",
    "spacecache": "#e91e63",
    "myopic": "#3498db",
    "otcp": "#2ecc71",
}


def ensure_effective_delay(graph: nx.Graph) -> nx.Graph:
    for u, v, edge in graph.edges(data=True):
        if "eff_delay" not in edge:
            delay = float(edge.get("delay", 0.0))
            edge["eff_delay"] = delay
        if "bw" not in edge:
            edge["bw"] = 1.0
    return graph


def build_snapshot_view(graph: nx.Graph, type_map: dict[str, str], cache_nodes=None, all_reachable=None) -> dict[str, Any]:
    graph = ensure_effective_delay(graph)
    if cache_nodes is None:
        cache_nodes = get_cache_nodes(graph, type_map)
    if all_reachable is None:
        all_reachable = get_all_reachable_sats(graph, type_map)
    return {
        "G": graph,
        "type_map": type_map,
        "cache_nodes": set(cache_nodes),
        "all_reachable": set(all_reachable),
    }


def initial_runtime_state(method: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "tracker": PopularityTracker(),
        "placement": {},
    }
    if method == "lce_lru":
        state["lce_mgr"] = LCELRUManager()
    if method == "madrl":
        state["madrl_mgr"] = MaDRLManager()
    return state


def copy_placement(placement: dict[str, set[int]] | dict[str, list[int]]) -> dict[str, set[int]]:
    copied: dict[str, set[int]] = {}
    for node_id, contents in placement.items():
        copied[node_id] = {int(item) for item in contents}
    return copied


def request_batch_scores(request_batch: list[tuple[str, int]]) -> dict[int, float]:
    demand = Counter()
    for _, content_id in request_batch:
        demand[int(content_id)] += 1
    return {content_id: float(score) for content_id, score in demand.items()}


def cold_start_zipf_prior() -> dict[int, float]:
    ranks = np.arange(1, cfg.CONTENT_CATALOG_SIZE + 1, dtype=float)
    weights = np.power(ranks, -cfg.ZIPF_ALPHA)
    weights /= weights.sum()
    return {int(content_id): float(weight) for content_id, weight in enumerate(weights)}


def _nearest_cache_node(snapshot: dict[str, Any], requester: str) -> str | None:
    graph = snapshot["G"]
    if not graph.has_node(requester):
        return None

    best_node = None
    best_cost = float("inf")
    for node_id in sorted(snapshot["all_reachable"]):
        if not graph.has_node(node_id):
            continue
        try:
            cost = float(nx.dijkstra_path_length(graph, requester, node_id, weight="eff_delay"))
        except nx.NetworkXNoPath:
            continue
        if cost < best_cost or (cost == best_cost and (best_node is None or node_id < best_node)):
            best_node = node_id
            best_cost = cost
    return best_node


def _assign_requested_content_locally(
    snapshot: dict[str, Any],
    placement: dict[str, set[int]],
    request_batch: list[tuple[str, int]],
) -> dict[str, set[int]]:
    if not request_batch:
        return placement

    scores = request_batch_scores(request_batch)
    requesters_by_content: dict[int, list[str]] = {}
    for requester, content_id in request_batch:
        requesters_by_content.setdefault(int(content_id), []).append(requester)

    adjusted = copy_placement(placement)
    for content_id, requesters in sorted(requesters_by_content.items(), key=lambda item: (-scores.get(item[0], 0.0), item[0])):
        preferred_nodes = []
        seen = set()
        for requester in sorted(set(requesters)):
            node_id = _nearest_cache_node(snapshot, requester)
            if node_id is None or node_id in seen:
                continue
            preferred_nodes.append(node_id)
            seen.add(node_id)

        for node_id in preferred_nodes:
            contents = adjusted.setdefault(node_id, set())
            if content_id in contents:
                continue
            if len(contents) >= cfg.CACHE_CAPACITY:
                victim = min(
                    contents,
                    key=lambda existing: (
                        1 if scores.get(existing, 0.0) > 0.0 else 0,
                        scores.get(existing, 0.0),
                        existing,
                    ),
                )
                contents.remove(victim)
            contents.add(content_id)
    return adjusted


def solve_method_placement(
    method: str,
    step_index: int,
    future_snapshots: list[dict[str, Any]],
    runtime_state: dict[str, Any],
    request_batch: list[tuple[str, int]] | None = None,
) -> tuple[dict[str, set[int]], dict[str, float]]:
    current = future_snapshots[0]
    tracker = runtime_state["tracker"]
    pop_scores = dict(tracker.scores)
    placement = runtime_state["placement"]

    start = time.perf_counter()
    if method == "otcp":
        if request_batch:
            for content_id, score in request_batch_scores(request_batch).items():
                pop_scores[content_id] = pop_scores.get(content_id, 0.0) + score
        if not pop_scores:
            pop_scores = cold_start_zipf_prior()
        planning_views = [
            {
                "G": snapshot["G"],
                "cache_nodes": snapshot["all_reachable"],
                "type_map": snapshot["type_map"],
            }
            for snapshot in future_snapshots
        ]
        placement, _ = solve_olcp(planning_views, placement, pop_scores)
        placement = _assign_requested_content_locally(current, placement, request_batch or [])
    elif method == "myopic":
        placement, _ = solve_myopic(
            {
                "G": current["G"],
                "cache_nodes": current["all_reachable"],
                "type_map": current["type_map"],
            },
            placement,
            pop_scores,
        )
    elif method == "greedy":
        if step_index % cfg.GREEDY_REFRESH_INTERVAL == 0 or not placement:
            placement = greedy_placement(sorted(current["cache_nodes"]), tracker)
    elif method == "submod":
        if step_index % cfg.GREEDY_REFRESH_INTERVAL == 0 or not placement:
            placement = submodular_greedy_placement(current["G"], current["all_reachable"], current["type_map"], pop_scores)
    elif method == "spacecache":
        if step_index % cfg.GREEDY_REFRESH_INTERVAL == 0 or not placement:
            placement = spacecache_placement(current["G"], current["all_reachable"], current["type_map"], pop_scores)
    elif method == "madrl":
        placement = runtime_state["madrl_mgr"].decide_placement(sorted(current["all_reachable"]), pop_scores)
    elif method == "lce_lru":
        placement = {
            node_id: store.contents()
            for node_id, store in runtime_state["lce_mgr"]._stores.items()
        }
    elif method == "nocache":
        placement = {}
    else:
        raise RuntimeError(f"Unhandled method: {method}")

    solve_time_ms = (time.perf_counter() - start) * 1000.0
    copied = copy_placement(placement)
    runtime_state["placement"] = copied
    return copied, {"solve_time_ms": float(solve_time_ms)}


def _best_serving_node(graph: nx.Graph, requester: str, placement: dict[str, set[int]], content_id: int):
    if not graph.has_node(requester):
        return None

    best_delay = float("inf")
    best_node = None
    for node_id, cached_items in placement.items():
        if content_id not in cached_items or not graph.has_node(node_id):
            continue
        try:
            path = nx.dijkstra_path(graph, requester, node_id, weight="eff_delay")
            delay = sum(graph[path[index]][path[index + 1]]["eff_delay"] for index in range(len(path) - 1))
        except nx.NetworkXNoPath:
            continue
        if delay < best_delay:
            best_delay = delay
            best_node = node_id
    return best_node


def _best_origin_path(graph: nx.Graph, requester: str, type_map: dict[str, str]):
    best_path = None
    best_delay = float("inf")
    for node_id, node_type in type_map.items():
        if node_type != "GS" or not graph.has_node(node_id):
            continue
        try:
            path = nx.dijkstra_path(graph, requester, node_id, weight="eff_delay")
            delay = sum(graph[path[index]][path[index + 1]]["eff_delay"] for index in range(len(path) - 1))
        except nx.NetworkXNoPath:
            continue
        if delay < best_delay:
            best_path = path
            best_delay = delay
    return best_path, best_delay


def resolve_request(
    method: str,
    graph: nx.Graph,
    requester: str,
    content_id: int,
    type_map: dict[str, str],
    placement: dict[str, set[int]],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    graph = ensure_effective_delay(graph)
    if not graph.has_node(requester):
        return {"success": False, "path": None, "delay_ms": None, "traffic_mb": None, "hit": False}

    transfer_cache = (cfg.CONTENT_SIZE_BITS / (cfg.CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (cfg.CONTENT_SIZE_BITS / (cfg.GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    if method == "lce_lru":
        manager = runtime_state["lce_mgr"]
        origin_path, origin_delay = _best_origin_path(graph, requester, type_map)
        if origin_path is None:
            return {"success": False, "path": None, "delay_ms": None, "traffic_mb": None, "hit": False}

        for index, node_id in enumerate(origin_path):
            if type_map.get(node_id) == "SAT" and manager._get(node_id).has(content_id):
                sub_path = origin_path[:index + 1]
                delay = sum(graph[sub_path[i]][sub_path[i + 1]]["eff_delay"] for i in range(len(sub_path) - 1)) + transfer_cache
                traffic = cfg.CONTENT_SIZE_MB * max(1, len(sub_path) - 1)
                return {"success": True, "path": sub_path, "delay_ms": delay, "traffic_mb": traffic, "hit": True}

        total_delay = 2 * origin_delay + transfer_origin
        traffic = cfg.CONTENT_SIZE_MB * max(1, len(origin_path) - 1)
        for node_id in origin_path:
            if type_map.get(node_id) == "SAT":
                manager._get(node_id).put(content_id)
        return {"success": True, "path": origin_path, "delay_ms": total_delay, "traffic_mb": traffic, "hit": False}

    serving_node = _best_serving_node(graph, requester, placement, content_id)
    if serving_node is not None:
        try:
            hit_path = nx.dijkstra_path(graph, requester, serving_node, weight="eff_delay")
        except nx.NetworkXNoPath:
            hit_path = None
        if hit_path is not None:
            hit_delay = sum(graph[hit_path[i]][hit_path[i + 1]]["eff_delay"] for i in range(len(hit_path) - 1)) + transfer_cache
            traffic = cfg.CONTENT_SIZE_MB * max(1, len(hit_path) - 1)
            return {"success": True, "path": hit_path, "delay_ms": hit_delay, "traffic_mb": traffic, "hit": True}

    origin_path, origin_delay = _best_origin_path(graph, requester, type_map)
    if origin_path is None:
        return {"success": False, "path": None, "delay_ms": None, "traffic_mb": None, "hit": False}

    total_delay = 2 * origin_delay + transfer_origin
    traffic = cfg.CONTENT_SIZE_MB * max(1, len(origin_path) - 1)
    return {"success": True, "path": origin_path, "delay_ms": total_delay, "traffic_mb": traffic, "hit": False}


def advance_runtime_state(
    method: str,
    current_snapshot: dict[str, Any],
    request_batch: list[tuple[str, int]],
    placement: dict[str, set[int]],
    runtime_state: dict[str, Any],
) -> None:
    tracker = runtime_state["tracker"]

    if method == "madrl":
        hit_counts = {node_id: 0 for node_id in current_snapshot["all_reachable"]}
        for requester, content_id in request_batch:
            serving_node = _best_serving_node(current_snapshot["G"], requester, placement, content_id)
            if serving_node is not None:
                hit_counts[serving_node] = hit_counts.get(serving_node, 0) + 1
        runtime_state["madrl_mgr"].feedback(sorted(current_snapshot["all_reachable"]), hit_counts, tracker.scores)

    tracker.decay_all()
    for _, content_id in request_batch:
        tracker.record(content_id)

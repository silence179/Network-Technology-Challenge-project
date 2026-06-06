import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from code.config import DISCOUNT_FACTOR as OTCP_DISCOUNT_FACTOR
from code.config import LOOKAHEAD_HORIZON as OTCP_LOOKAHEAD_HORIZON
from code.olcp.solver import solve_olcp


MAX_LINK_RANGE = 5000 * 1000
MIN_ELEVATION_DEG = 10.0
SPEED_OF_LIGHT = 3e8
TOPO_HASH_INTERVAL = 100
DEFAULT_MAX_ROUTE_HOPS = 16
OTCP_MAX_CANDIDATE_SATS = 8
OTCP_ROUTE_MARGIN = 1.1
OTCP_POPULARITY_DECAY = 0.95
OTCP_UNREACHABLE_COST = 1e5

FLOWS = {
    "CTRL_FLOW": {
        "src": "GS_01",
        "priority": "HIGH",
        "base_bw_mbps": 0.01,
        "dst_cidr": "10.99.0.0/24",
    },
    "VIDEO_FLOW_UAV_01": {
        "src": "UAV_01",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 180,
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.1.1/32",
    },
    "VIDEO_FLOW_UAV_02": {
        "src": "UAV_02",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 300,
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.2.1/32",
    },
    "VIDEO_FLOW_UAV_03": {
        "src": "UAV_03",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 360,
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.3.1/32",
    },
}

TRAFFIC_MODEL_LEGACY = "legacy"
TRAFFIC_MODEL_PERIODIC_PHOTO = "photo"
TRAFFIC_MODEL_S4_COMPAT = "s4"
DEFAULT_PHOTO_INTERVAL_S = 10.0
DEFAULT_PHOTO_SIZE_MB = 12.0
DEFAULT_PHOTO_PRIORITY = "NORMAL"
S4_COMPAT_MAX_HOPS = {
    "s4_gs_to_uav": 2,
    "s4_uav_to_gs": 3,
    "s4_uav_to_uav": 2,
}
S4_COMPAT_MAX_DELAY_MS = {
    "s4_gs_to_uav": 8.0,
    "s4_uav_to_gs": 8.0,
    "s4_uav_to_uav": 6.0,
}
S4_COMPAT_MIN_BW_MBPS = {
    "s4_gs_to_uav": 10.0,
    "s4_uav_to_gs": 10.0,
    "s4_uav_to_uav": 10.0,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TopologyCache:
    def __init__(self):
        self.last_topology_hash = None
        self.last_topology = None

    def get_hash(self, links):
        if not links:
            return None
        link_str = "|".join(
            f"{link['src']}-{link['dst']}"
            for link in sorted(links, key=lambda item: (item["src"], item["dst"]))
            if link.get("status", "UP") == "UP"
        )
        return hashlib.md5(link_str.encode("utf-8")).hexdigest()

    def is_topology_changed(self, links):
        new_hash = self.get_hash(links)
        changed = new_hash != self.last_topology_hash
        if changed:
            self.last_topology_hash = new_hash
        return changed

    def cache_topology(self, links):
        self.last_topology = links


@dataclass
class RoutePlan:
    path: list
    algo: str
    estimated_delay_ms: float
    bottleneck_bw_mbps: float
    notes: str = ""


@dataclass
class RoutingContext:
    time_ms: int
    topology_hash: str
    node_positions: dict
    node_ip_map: dict
    graph_delay: nx.Graph
    graph_high: nx.Graph
    graph_normal: nx.Graph
    active_edge_lookup: dict
    node_loads: dict
    type_map: dict | None = None
    future_snapshots: list | None = None
    flow_requests: list | None = None


class SimulationStats:
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.step_count = 0
        self.topology_reused_steps = 0
        self.topology_changed_steps = 0
        self.total_flow_requests = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.total_rules = 0
        self.sum_path_delay_ms = 0.0
        self.sum_hops = 0
        self.sum_bottleneck_bw = 0.0
        self.total_active_links = 0
        self.peak_active_links = 0
        self.topology_compute_time_ms = 0.0
        self.routing_compute_time_ms = 0.0

    def record_step(self, active_links_count, reused_topology, topology_changed):
        self.step_count += 1
        self.total_active_links += active_links_count
        self.peak_active_links = max(self.peak_active_links, active_links_count)
        if reused_topology:
            self.topology_reused_steps += 1
        if topology_changed:
            self.topology_changed_steps += 1

    def record_success(self, plan):
        self.successful_routes += 1
        self.total_rules += 1
        self.sum_path_delay_ms += plan.estimated_delay_ms
        self.sum_hops += max(0, len(plan.path) - 1)
        self.sum_bottleneck_bw += plan.bottleneck_bw_mbps

    def record_failure(self):
        self.failed_routes += 1

    def to_dict(self, runtime_seconds, sat_dir, max_steps):
        avg_active_links = self.total_active_links / self.step_count if self.step_count else 0.0
        avg_path_delay = self.sum_path_delay_ms / self.successful_routes if self.successful_routes else 0.0
        avg_hops = self.sum_hops / self.successful_routes if self.successful_routes else 0.0
        avg_bottleneck_bw = self.sum_bottleneck_bw / self.successful_routes if self.successful_routes else 0.0
        success_rate = (self.successful_routes / self.total_flow_requests * 100.0) if self.total_flow_requests else 0.0
        reuse_rate = (self.topology_reused_steps / self.step_count * 100.0) if self.step_count else 0.0
        return {
            "Algorithm": self.algorithm_name,
            "Satellite Trace": sat_dir,
            "Max Steps": max_steps if max_steps is not None else "ALL",
            "Runtime (s)": round(runtime_seconds, 3),
            "Topology Compute Time (ms)": round(self.topology_compute_time_ms, 3),
            "Routing Compute Time (ms)": round(self.routing_compute_time_ms, 3),
            "Time Steps": self.step_count,
            "Flow Requests": self.total_flow_requests,
            "Successful Routes": self.successful_routes,
            "Failed Routes": self.failed_routes,
            "Route Success Rate (%)": round(success_rate, 3),
            "Total Rules": self.total_rules,
            "Avg Path Delay (ms)": round(avg_path_delay, 3),
            "Avg Hop Count": round(avg_hops, 3),
            "Avg Bottleneck BW (Mbps)": round(avg_bottleneck_bw, 3),
            "Avg Active Links": round(avg_active_links, 3),
            "Peak Active Links": self.peak_active_links,
            "Topology Changed Steps": self.topology_changed_steps,
            "Topology Reuse Rate (%)": round(reuse_rate, 3),
        }


def load_and_merge_traces(sat_dir="traces/sat_trace", uav_file="traces/uav_trace_full.csv"):
    def resolve_uav_source(path):
        candidates = [path]
        basename = os.path.basename(path)
        parent_dir = os.path.dirname(path)
        nested_candidate = os.path.join(parent_dir, "uav_trace", basename)
        flat_candidate = os.path.join(parent_dir, basename)

        for candidate in (nested_candidate, flat_candidate):
            if candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if candidate and (os.path.exists(candidate) or os.path.isdir(candidate)):
                return candidate
        return path

    sat_files = sorted(glob.glob(os.path.join(sat_dir, "*.csv")))
    sat_frames = [pd.read_csv(file_path) for file_path in sat_files]
    df_sat = pd.concat(sat_frames, ignore_index=True) if sat_frames else pd.DataFrame()
    resolved_uav_source = resolve_uav_source(uav_file)
    if os.path.isdir(resolved_uav_source):
        uav_files = sorted(glob.glob(os.path.join(resolved_uav_source, "*.csv")))
        uav_frames = [pd.read_csv(file_path) for file_path in uav_files]
        df_uav = pd.concat(uav_frames, ignore_index=True) if uav_frames else pd.DataFrame()
    else:
        df_uav = pd.read_csv(resolved_uav_source) if os.path.exists(resolved_uav_source) else pd.DataFrame()
    timelines = sorted(df_uav["time_ms"].unique()) if not df_uav.empty else []
    return df_sat, df_uav, timelines


def get_nodes_at_timestamp(df_sat, df_uav, target_time_ms):
    cols = ["node_id", "type", "ecef_x", "ecef_y", "ecef_z", "ip"]
    uav_current = df_uav[df_uav["time_ms"] == target_time_ms]
    sat_time_key = (target_time_ms // 1000) * 1000
    sat_current = df_sat[df_sat["time_ms"] == sat_time_key]
    if sat_current.empty and uav_current.empty:
        return pd.DataFrame(columns=cols)
    return pd.concat([sat_current[cols], uav_current[cols]], ignore_index=True)


def calculate_delay(dist_m):
    return (dist_m / SPEED_OF_LIGHT) * 1000


def calculate_jitter(delay_ms):
    return delay_ms * 0.1


def calculate_bandwidth(type_a, type_b):
    types = {type_a, type_b}
    if "GS" in types and "UAV" in types:
        return 0
    if "UAV" in types and "SAT" in types:
        return 20
    if "SAT" in types and "GS" in types:
        return 20
    if "SAT" in types:
        return 100
    return 10


def calculate_bdp_queue(bw_mbps, delay_ms):
    queue = int((bw_mbps * 1e6) * (delay_ms * 2 * 1e-3) / 12000)
    return max(10, queue)


def calculate_elevation(pos_a, pos_b):
    vector_a = np.array(pos_a)
    vector_ab = np.array(pos_b) - np.array(pos_a)
    norm_a = np.linalg.norm(vector_a)
    norm_ab = np.linalg.norm(vector_ab)
    if norm_a == 0 or norm_ab == 0:
        return 90.0
    cos_theta = np.dot(vector_a, vector_ab) / (norm_a * norm_ab)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta_rad = np.arccos(cos_theta)
    return 90.0 - math.degrees(theta_rad)


def compute_topology(nodes_df, time_ms):
    links = []
    if len(nodes_df) < 2:
        return links

    coords = nodes_df[["ecef_x", "ecef_y", "ecef_z"]].values
    node_ids = nodes_df["node_id"].values
    node_types = nodes_df["type"].values

    tree = cKDTree(coords)
    distances, neighbor_indices = tree.query(coords, k=20, distance_upper_bound=MAX_LINK_RANGE)
    processed_pairs = set()

    for source_index in range(len(node_ids)):
        for neighbor_slot, target_index in enumerate(neighbor_indices[source_index]):
            if distances[source_index][neighbor_slot] == float("inf") or source_index == target_index:
                continue

            source_id = node_ids[source_index]
            target_id = node_ids[target_index]
            pair_key = tuple(sorted((source_id, target_id)))
            if pair_key in processed_pairs:
                continue

            type_a = node_types[source_index]
            type_b = node_types[target_index]
            is_sat_a = type_a == "SAT"
            is_sat_b = type_b == "SAT"

            if is_sat_a != is_sat_b:
                sat_index = source_index if is_sat_a else target_index
                ground_index = target_index if is_sat_a else source_index
                elevation = calculate_elevation(coords[ground_index], coords[sat_index])
                if elevation < MIN_ELEVATION_DEG:
                    continue

            dist_m = distances[source_index][neighbor_slot]
            delay_ms = calculate_delay(dist_m)
            bandwidth = calculate_bandwidth(type_a, type_b)
            if bandwidth == 0:
                continue

            links.append(
                {
                    "time_ms": time_ms,
                    "src": source_id,
                    "dst": target_id,
                    "direction": "BIDIR",
                    "distance_km": round(dist_m / 1000.0, 3),
                    "delay_ms": round(delay_ms, 3),
                    "jitter_ms": round(calculate_jitter(delay_ms), 3),
                    "loss_pct": 0.0,
                    "bw_mbps": bandwidth,
                    "max_queue_pkt": calculate_bdp_queue(bandwidth, delay_ms),
                    "type": f"{type_a}-{type_b}",
                    "status": "UP",
                    "lifetime_ms": 60000,
                }
            )
            processed_pairs.add(pair_key)

    return links


def build_graph_for_priority(links, priority):
    graph = nx.Graph()
    for link in links:
        delay = link["delay_ms"]
        lifetime_sec = max(link["lifetime_ms"] / 1000.0, 0.1)
        if priority == "HIGH":
            stability_penalty = 1000.0 / lifetime_sec
            weight = delay * 0.1 + stability_penalty
        elif priority == "NORMAL":
            stability_penalty = 50.0 / lifetime_sec
            weight = delay + stability_penalty
        else:
            weight = delay

        graph.add_edge(
            link["src"],
            link["dst"],
            weight=weight,
            delay_ms=link["delay_ms"],
            bw_mbps=link["bw_mbps"],
            lifetime_ms=link["lifetime_ms"],
            distance_km=link["distance_km"],
        )
    return graph


_bandwidth_cache = {}


def get_current_bandwidth(flow_name, flow_config, current_time_ms):
    cache_key = (flow_name, current_time_ms)
    if cache_key in _bandwidth_cache:
        return _bandwidth_cache[cache_key]

    time_seconds = current_time_ms / 1000.0
    if "burst_time_s" in flow_config and time_seconds >= flow_config["burst_time_s"]:
        base_bw = flow_config["burst_bw_mbps"]
    else:
        base_bw = flow_config["base_bw_mbps"]

    rng = random.Random(int(time_seconds * 10))
    fluctuation = base_bw * rng.uniform(-0.05, 0.05)
    current_bw = round(max(0.01, base_bw + fluctuation), 2)
    _bandwidth_cache[cache_key] = current_bw
    return current_bw


def build_node_ip_map(df_sat, df_uav):
    node_ip_map = {}
    if not df_sat.empty:
        for _, row in df_sat[["node_id", "ip"]].drop_duplicates().iterrows():
            node_ip_map[row["node_id"]] = row["ip"]
    if not df_uav.empty:
        for _, row in df_uav[["node_id", "ip"]].drop_duplicates().iterrows():
            node_ip_map[row["node_id"]] = row["ip"]
    return node_ip_map


def build_periodic_photo_flow_config(uav_id, photo_size_mb, photo_interval_s):
    req_bw_mbps = round((photo_size_mb * 8.0) / max(photo_interval_s, 0.1), 3)
    return {
        "src": uav_id,
        "priority": DEFAULT_PHOTO_PRIORITY,
        "base_bw_mbps": req_bw_mbps,
        "dst_cidr": "0.0.0.0/0",
        "traffic_type": "periodic_photo",
        "photo_size_mb": float(photo_size_mb),
        "photo_interval_s": float(photo_interval_s),
        "flow_period_ms": int(round(photo_interval_s * 1000.0)),
    }


def build_s4_compat_flow_requests(active_nodes):
    requests = []
    active_node_set = set(active_nodes)
    active_uavs = sorted(node_id for node_id in active_node_set if str(node_id).startswith("UAV_"))
    if not active_uavs:
        return requests

    gs_id = "GS_01"
    if gs_id in active_node_set:
        for uav_id in active_uavs:
            requests.append(
                {
                    "flow_name": f"S4_GS_TO_{uav_id}",
                    "src": gs_id,
                    "dst": uav_id,
                    "config": {
                        "src": gs_id,
                        "priority": "NORMAL",
                        "base_bw_mbps": 10.0,
                        "dst_cidr": "0.0.0.0/0",
                        "traffic_type": "s4_gs_to_uav",
                    },
                }
            )
            requests.append(
                {
                    "flow_name": f"S4_{uav_id}_TO_GS",
                    "src": uav_id,
                    "dst": gs_id,
                    "config": {
                        "src": uav_id,
                        "priority": "HIGH",
                        "base_bw_mbps": 10.0,
                        "dst_cidr": "0.0.0.0/0",
                        "traffic_type": "s4_uav_to_gs",
                    },
                }
            )

    for source_uav in active_uavs:
        for target_uav in active_uavs:
            if source_uav == target_uav:
                continue
            requests.append(
                {
                    "flow_name": f"S4_{source_uav}_TO_{target_uav}",
                    "src": source_uav,
                    "dst": target_uav,
                    "config": {
                        "src": source_uav,
                        "priority": "NORMAL",
                        "base_bw_mbps": 2.0,
                        "dst_cidr": "0.0.0.0/0",
                        "traffic_type": "s4_uav_to_uav",
                    },
                }
            )

    return requests


def build_flow_requests(
    active_nodes,
    current_time_ms,
    traffic_model=TRAFFIC_MODEL_LEGACY,
    photo_interval_s=DEFAULT_PHOTO_INTERVAL_S,
    photo_size_mb=DEFAULT_PHOTO_SIZE_MB,
    include_ctrl_flow=True,
    emit_photo_every_step=False,
):
    requests = []
    active_node_set = set(active_nodes)

    if traffic_model == TRAFFIC_MODEL_S4_COMPAT:
        return build_s4_compat_flow_requests(active_nodes)

    if include_ctrl_flow and FLOWS["CTRL_FLOW"]["src"] in active_node_set:
        active_uavs = sorted(node_id for node_id in active_node_set if node_id.startswith("UAV_"))
        for target_uav in active_uavs:
            requests.append(
                {
                    "flow_name": "CTRL_FLOW",
                    "src": FLOWS["CTRL_FLOW"]["src"],
                    "dst": target_uav,
                    "config": FLOWS["CTRL_FLOW"],
                }
            )

    if traffic_model == TRAFFIC_MODEL_PERIODIC_PHOTO:
        interval_ms = max(1, int(round(photo_interval_s * 1000.0)))
        if not emit_photo_every_step and current_time_ms % interval_ms != 0:
            return requests

        if "GS_01" in active_node_set:
            active_uavs = sorted(node_id for node_id in active_node_set if node_id.startswith("UAV_"))
            for source_uav in active_uavs:
                requests.append(
                    {
                        "flow_name": f"PHOTO_FLOW_{source_uav}",
                        "src": source_uav,
                        "dst": "GS_01",
                        "config": build_periodic_photo_flow_config(source_uav, photo_size_mb, photo_interval_s),
                    }
                )
        return requests

    if "GS_01" in active_node_set:
        for flow_name, flow_config in FLOWS.items():
            if flow_name.startswith("VIDEO_FLOW_") and flow_config["src"] in active_node_set:
                requests.append(
                    {
                        "flow_name": flow_name,
                        "src": flow_config["src"],
                        "dst": "GS_01",
                        "config": flow_config,
                    }
                )
    return requests


def build_type_map(nodes_df):
    if nodes_df.empty:
        return {}
    return {row["node_id"]: row["type"] for _, row in nodes_df[["node_id", "type"]].iterrows()}


def build_effective_delay_graph(active_links, nodes_df):
    graph = nx.Graph()
    if not nodes_df.empty:
        for node_id in nodes_df["node_id"].tolist():
            graph.add_node(node_id)
    for link in active_links:
        graph.add_edge(
            link["src"],
            link["dst"],
            eff_delay=float(link.get("delay_ms", 0.0)),
            delay_ms=float(link.get("delay_ms", 0.0)),
            bw=float(link.get("bw_mbps", 0.0)),
        )
    return graph


def select_otcp_candidate_sats(graph, type_map, max_n=OTCP_MAX_CANDIDATE_SATS):
    uav_nodes = [node_id for node_id, node_type in type_map.items() if node_type == "UAV" and graph.has_node(node_id)]
    sat_nodes = [node_id for node_id, node_type in type_map.items() if node_type == "SAT" and graph.has_node(node_id)]
    if not uav_nodes or not sat_nodes:
        return set()

    candidates = set()
    for uav_id in uav_nodes:
        for neighbor in graph.neighbors(uav_id):
            if type_map.get(neighbor) == "SAT":
                candidates.add(neighbor)

    if not candidates:
        return set()

    ranked = []
    for sat_id in candidates:
        best_delay = min(
            (
                float(graph[uav_id][sat_id].get("eff_delay", 0.0))
                for uav_id in uav_nodes
                if graph.has_edge(uav_id, sat_id)
            ),
            default=float("inf"),
        )
        ranked.append((best_delay, sat_id))

    ranked.sort(key=lambda item: item[0])
    return {sat_id for _, sat_id in ranked[:max_n]}


def pair_key(node_a, node_b):
    return (node_a, node_b) if node_a < node_b else (node_b, node_a)


def make_edge_lookup(graph):
    lookup = {}
    for source_id, target_id, attrs in graph.edges(data=True):
        lookup[pair_key(source_id, target_id)] = dict(attrs)
    return lookup


def summarize_path(path, edge_lookup):
    total_delay = 0.0
    bottleneck = float("inf")
    for source_id, target_id in zip(path, path[1:]):
        attrs = edge_lookup.get(pair_key(source_id, target_id))
        if attrs is None:
            return None
        total_delay += float(attrs.get("delay_ms", 0.0))
        bottleneck = min(bottleneck, float(attrs.get("bw_mbps", 0.0)))
    if bottleneck == float("inf"):
        bottleneck = 0.0
    return total_delay, bottleneck


def make_rules(flow_request, plan, time_ms, node_ip_map):
    destination = flow_request["dst"]
    flow_name = flow_request["flow_name"]
    flow_config = flow_request["config"]
    rules = []
    for current_node, next_hop in zip(plan.path[:-1], plan.path[1:]):
        rules.append(
            {
                "time_ms": int(time_ms),
                "node": current_node,
                "dst_cidr": f"{node_ip_map.get(destination, '0.0.0.0')}/32",
                "action": "replace",
                "next_hop": next_hop,
                "next_hop_ip": node_ip_map.get(next_hop, "0.0.0.0"),
                "algo": plan.algo,
                "req_bw_mbps": get_current_bandwidth(flow_name, flow_config, time_ms),
                "traffic_type": flow_config.get("traffic_type", "continuous"),
                "photo_size_mb": flow_config.get("photo_size_mb"),
                "photo_interval_s": flow_config.get("photo_interval_s"),
                "debug_info": plan.notes,
            }
        )
    return rules


def should_export_plan(flow_request, plan):
    traffic_type = flow_request["config"].get("traffic_type")
    if not traffic_type or not str(traffic_type).startswith("s4_"):
        return True

    hop_count = max(0, len(plan.path) - 1)
    max_hops = S4_COMPAT_MAX_HOPS.get(traffic_type)
    max_delay_ms = S4_COMPAT_MAX_DELAY_MS.get(traffic_type)
    min_bw_mbps = S4_COMPAT_MIN_BW_MBPS.get(traffic_type)

    if max_hops is not None and hop_count > max_hops:
        return False
    if max_delay_ms is not None and plan.estimated_delay_ms > max_delay_ms:
        return False
    if min_bw_mbps is not None and plan.bottleneck_bw_mbps < min_bw_mbps:
        return False
    return True


class BaseRouter:
    algorithm_name = "Base"
    requires_future_snapshots = False

    def begin_step(self, context):
        return None

    def plan_route(self, flow_request, context):
        raise NotImplementedError

    def _plan_from_path(self, path, edge_lookup, notes=""):
        summary = summarize_path(path, edge_lookup)
        if summary is None:
            return None
        total_delay, bottleneck_bw = summary
        return RoutePlan(
            path=path,
            algo=self.algorithm_name,
            estimated_delay_ms=round(total_delay, 3),
            bottleneck_bw_mbps=round(bottleneck_bw, 3),
            notes=notes,
        )


class OptimizedRouter(BaseRouter):
    algorithm_name = "Optimized"

    def plan_route(self, flow_request, context):
        priority = flow_request["config"]["priority"]
        graph = context.graph_high if priority == "HIGH" else context.graph_normal
        src = flow_request["src"]
        dst = flow_request["dst"]
        if not graph.has_node(src) or not graph.has_node(dst):
            return None
        try:
            path = nx.shortest_path(graph, src, dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return self._plan_from_path(path, context.active_edge_lookup, f"priority={priority}")


class HypatiaRouter(BaseRouter):
    algorithm_name = "Hypatia"

    def plan_route(self, flow_request, context):
        src = flow_request["src"]
        dst = flow_request["dst"]
        if not context.graph_delay.has_node(src) or not context.graph_delay.has_node(dst):
            return None
        try:
            path = nx.shortest_path(context.graph_delay, src, dst, weight="delay_ms")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return self._plan_from_path(path, context.active_edge_lookup, "global_shortest_path")


class LSRRouter(BaseRouter):
    algorithm_name = "LSR"

    def plan_route(self, flow_request, context):
        src = flow_request["src"]
        dst = flow_request["dst"]
        if src not in context.node_positions or dst not in context.node_positions:
            return None

        path = [src]
        visited = {src}
        current = src

        for _ in range(DEFAULT_MAX_ROUTE_HOPS):
            if current == dst:
                return self._plan_from_path(path, context.active_edge_lookup, "greedy_local_state")
            if not context.graph_delay.has_node(current):
                return None

            current_distance = np.linalg.norm(context.node_positions[current] - context.node_positions[dst])
            best_neighbor = None
            best_score = float("inf")

            for neighbor in context.graph_delay.neighbors(current):
                if neighbor in visited:
                    continue
                edge_attrs = context.graph_delay[current][neighbor]
                neighbor_distance = np.linalg.norm(context.node_positions[neighbor] - context.node_positions[dst])
                queue_penalty = context.node_loads[neighbor] * 50000.0
                delay_penalty = float(edge_attrs.get("delay_ms", 0.0)) * 10000.0
                score = neighbor_distance + queue_penalty + delay_penalty
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor

            if best_neighbor is None:
                return None

            next_distance = np.linalg.norm(context.node_positions[best_neighbor] - context.node_positions[dst])
            if next_distance >= current_distance and len(path) > 1:
                return None

            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        return None


class MADRLRouter(BaseRouter):
    algorithm_name = "MA-DRL"

    def __init__(self):
        self.weight_1 = np.array(
            [
                [1.6, -0.9, 0.4, 0.8, 0.3, -0.5, 0.7, 0.2],
                [-1.1, 0.7, -0.8, 0.1, 0.6, 0.5, -0.4, 0.3],
                [-0.6, -0.2, -0.9, 0.2, -0.5, 0.3, -0.1, 0.4],
                [0.8, 0.5, 0.2, -0.7, 0.9, -0.6, 0.3, -0.4],
                [0.7, -0.4, 0.6, 0.5, -0.1, 0.8, -0.7, 0.2],
                [0.2, 0.9, -0.1, 0.6, -0.3, 0.7, 0.5, -0.8],
            ],
            dtype=float,
        )
        self.bias_1 = np.array([0.1, -0.2, 0.3, 0.0, 0.15, -0.1, 0.05, 0.12], dtype=float)
        self.weight_2 = np.array([1.8, -0.9, 0.6, 1.2, 1.5, 0.9, -0.7, 0.4], dtype=float)

    def _score_neighbor(self, current, neighbor, dst, context):
        current_distance = np.linalg.norm(context.node_positions[current] - context.node_positions[dst])
        neighbor_distance = np.linalg.norm(context.node_positions[neighbor] - context.node_positions[dst])
        edge_attrs = context.graph_delay[current][neighbor]

        progress = (current_distance - neighbor_distance) / max(current_distance, 1.0)
        normalized_delay = float(edge_attrs.get("delay_ms", 0.0)) / 20.0
        normalized_bw = float(edge_attrs.get("bw_mbps", 0.0)) / 100.0
        normalized_load = context.node_loads[neighbor] / 10.0
        normalized_degree = context.graph_delay.degree(neighbor) / 10.0
        stability = float(edge_attrs.get("lifetime_ms", 0.0)) / 60000.0

        features = np.array(
            [progress, -normalized_delay, normalized_bw, -normalized_load, normalized_degree, stability],
            dtype=float,
        )
        hidden = np.tanh(features @ self.weight_1 + self.bias_1)
        return float(hidden @ self.weight_2)

    def plan_route(self, flow_request, context):
        src = flow_request["src"]
        dst = flow_request["dst"]
        if src not in context.node_positions or dst not in context.node_positions:
            return None

        path = [src]
        visited = {src}
        current = src

        for _ in range(DEFAULT_MAX_ROUTE_HOPS):
            if current == dst:
                return self._plan_from_path(path, context.active_edge_lookup, "policy_inference")
            if not context.graph_delay.has_node(current):
                return None

            candidates = []
            for neighbor in context.graph_delay.neighbors(current):
                if neighbor in visited:
                    continue
                score = self._score_neighbor(current, neighbor, dst, context)
                candidates.append((score, neighbor))

            if not candidates:
                return None

            candidates.sort(reverse=True)
            best_neighbor = candidates[0][1]
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        return None


class OTCPRouter(BaseRouter):
    algorithm_name = "OTCP"
    requires_future_snapshots = True

    def __init__(self):
        self.lookahead_horizon = OTCP_LOOKAHEAD_HORIZON
        self.discount_factor = OTCP_DISCOUNT_FACTOR
        self.cache_state = {}
        self.flow_id_map = {}
        self.flow_popularity = {}

    def _get_flow_id(self, flow_request):
        flow_key = (flow_request["flow_name"], flow_request["src"], flow_request["dst"])
        if flow_key not in self.flow_id_map:
            self.flow_id_map[flow_key] = len(self.flow_id_map)
        return self.flow_id_map[flow_key]

    def _direct_plan(self, flow_request, context):
        src = flow_request["src"]
        dst = flow_request["dst"]
        if not context.graph_delay.has_node(src) or not context.graph_delay.has_node(dst):
            return None
        try:
            path = nx.shortest_path(context.graph_delay, src, dst, weight="delay_ms")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return self._plan_from_path(path, context.active_edge_lookup, f"lookahead={self.lookahead_horizon};mode=direct")

    def _combine_paths(self, path_a, path_b, graph):
        combined = path_a + path_b[1:]
        if len(combined) != len(set(combined)):
            return None
        for source_id, target_id in zip(combined, combined[1:]):
            if not graph.has_edge(source_id, target_id):
                return None
        return combined

    def _future_cost(self, snapshots, src, dst, relay=None):
        total_cost = 0.0
        for tau, snapshot in enumerate(snapshots):
            weight = self.discount_factor ** tau
            graph = snapshot["G"]
            try:
                if relay is None:
                    cost = nx.dijkstra_path_length(graph, src, dst, weight="eff_delay")
                else:
                    first_leg = nx.dijkstra_path_length(graph, src, relay, weight="eff_delay")
                    second_leg = nx.dijkstra_path_length(graph, relay, dst, weight="eff_delay")
                    cost = first_leg + second_leg
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                cost = OTCP_UNREACHABLE_COST
            total_cost += weight * cost
        return total_cost

    def begin_step(self, context):
        for flow_id in list(self.flow_popularity):
            self.flow_popularity[flow_id] *= OTCP_POPULARITY_DECAY

        for flow_request in context.flow_requests or []:
            flow_id = self._get_flow_id(flow_request)
            self.flow_popularity[flow_id] = self.flow_popularity.get(flow_id, 0.0) + 1.0

        snapshots = context.future_snapshots or []
        if not snapshots:
            return

        popularity_scores = {
            flow_id: score
            for flow_id, score in self.flow_popularity.items()
            if flow_id < 100 and score > 1e-6
        }
        if not popularity_scores:
            return

        horizon = min(self.lookahead_horizon, max(len(snapshots) - 1, 0))
        self.cache_state, _ = solve_olcp(
            snapshots,
            self.cache_state,
            popularity_scores,
            horizon=horizon,
        )

    def plan_route(self, flow_request, context):
        direct_plan = self._direct_plan(flow_request, context)
        flow_id = self._get_flow_id(flow_request)
        src = flow_request["src"]
        dst = flow_request["dst"]
        candidate_relays = [
            sat_id
            for sat_id, content_ids in self.cache_state.items()
            if flow_id in content_ids and context.graph_delay.has_node(sat_id)
        ]
        if not candidate_relays:
            return direct_plan

        direct_future_cost = self._future_cost(context.future_snapshots or [], src, dst)
        best_plan = direct_plan
        best_benefit = 0.0
        direct_delay = direct_plan.estimated_delay_ms if direct_plan is not None else float("inf")

        for relay_id in candidate_relays:
            try:
                first_path = nx.shortest_path(context.graph_delay, src, relay_id, weight="delay_ms")
                second_path = nx.shortest_path(context.graph_delay, relay_id, dst, weight="delay_ms")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            combined_path = self._combine_paths(first_path, second_path, context.graph_delay)
            if combined_path is None or len(combined_path) < 2:
                continue

            summary = summarize_path(combined_path, context.active_edge_lookup)
            if summary is None:
                continue
            relay_delay, relay_bw = summary
            if direct_plan is not None and relay_delay > direct_delay * OTCP_ROUTE_MARGIN:
                continue

            relay_future_cost = self._future_cost(context.future_snapshots or [], src, dst, relay=relay_id)
            benefit = direct_future_cost - relay_future_cost
            if benefit <= best_benefit:
                continue

            best_benefit = benefit
            best_plan = RoutePlan(
                path=combined_path,
                algo=self.algorithm_name,
                estimated_delay_ms=round(relay_delay, 3),
                bottleneck_bw_mbps=round(relay_bw, 3),
                notes=f"lookahead={self.lookahead_horizon};relay={relay_id};future_gain={benefit:.2f}",
            )

        if best_plan is not None and best_plan is not direct_plan:
            return best_plan
        if direct_plan is not None:
            return direct_plan

        return None


class FTRLRouter(BaseRouter):
    algorithm_name = "FTRL"

    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.path_cache = {}
        self.cumulative_loss = {}

    def plan_route(self, flow_request, context):
        src = flow_request["src"]
        dst = flow_request["dst"]
        if not context.graph_delay.has_node(src) or not context.graph_delay.has_node(dst):
            return None

        cache_key = (context.topology_hash, src, dst)
        if cache_key not in self.path_cache:
            try:
                path_generator = nx.shortest_simple_paths(context.graph_delay, src, dst, weight="delay_ms")
                candidate_paths = list(itertools.islice(path_generator, 2))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            if not candidate_paths:
                return None
            self.path_cache[cache_key] = candidate_paths
            self.cumulative_loss.setdefault(cache_key, [0.0] * len(candidate_paths))

        candidate_paths = self.path_cache[cache_key]
        losses = self.cumulative_loss[cache_key]

        instant_losses = []
        for path in candidate_paths:
            summary = summarize_path(path, context.active_edge_lookup)
            if summary is None:
                instant_losses.append(9999.0)
                continue
            total_delay, _ = summary
            congestion = sum(context.node_loads[node_id] for node_id in path) / max(len(path), 1)
            instant_losses.append(total_delay / 10.0 + congestion)

        for index, loss_value in enumerate(instant_losses):
            losses[index] += loss_value

        weights = np.exp(-self.learning_rate * np.array(losses, dtype=float))
        selected_index = int(np.argmax(weights))
        selected_path = candidate_paths[selected_index]
        return self._plan_from_path(selected_path, context.active_edge_lookup, "online_two_path_selection")


class DTNCGRRouter(BaseRouter):
    algorithm_name = "DTN-CGR"

    def __init__(self, history_window=5):
        self.history_window = history_window
        self.contact_history = deque(maxlen=history_window)
        self.contact_graph = nx.Graph()
        self.contact_lookup = {}

    def begin_step(self, context):
        snapshot = []
        for source_id, target_id, attrs in context.graph_delay.edges(data=True):
            snapshot.append((source_id, target_id, dict(attrs)))
        self.contact_history.append(snapshot)

        edge_counts = defaultdict(int)
        delay_sums = defaultdict(float)
        bw_mins = {}
        graph = nx.Graph()

        for history_snapshot in self.contact_history:
            for source_id, target_id, attrs in history_snapshot:
                key = pair_key(source_id, target_id)
                edge_counts[key] += 1
                delay_sums[key] += float(attrs.get("delay_ms", 0.0))
                if key not in bw_mins:
                    bw_mins[key] = float(attrs.get("bw_mbps", 0.0))
                else:
                    bw_mins[key] = min(bw_mins[key], float(attrs.get("bw_mbps", 0.0)))

        for key, count in edge_counts.items():
            source_id, target_id = key
            availability = count / max(len(self.contact_history), 1)
            avg_delay = delay_sums[key] / count
            contact_weight = avg_delay + (1.0 - availability) * 150.0
            graph.add_edge(
                source_id,
                target_id,
                contact_weight=contact_weight,
                delay_ms=avg_delay,
                bw_mbps=bw_mins[key],
            )

        self.contact_graph = graph
        self.contact_lookup = make_edge_lookup(graph)

    def plan_route(self, flow_request, context):
        src = flow_request["src"]
        dst = flow_request["dst"]
        if not context.graph_delay.has_node(src) or src not in self.contact_graph or dst not in self.contact_graph:
            return None

        try:
            immediate_path = nx.shortest_path(context.graph_delay, src, dst, weight="delay_ms")
            return self._plan_from_path(immediate_path, context.active_edge_lookup, "current_contact")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        best_plan = None
        best_weight = float("inf")
        for neighbor in context.graph_delay.neighbors(src):
            try:
                predicted_path = nx.shortest_path(self.contact_graph, neighbor, dst, weight="contact_weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            full_path = [src] + predicted_path
            summary = summarize_path(full_path, self.contact_lookup)
            if summary is None:
                continue
            total_delay, bottleneck_bw = summary
            path_weight = total_delay + len(full_path) * 25.0
            if path_weight < best_weight:
                best_weight = path_weight
                best_plan = RoutePlan(
                    path=full_path,
                    algo=self.algorithm_name,
                    estimated_delay_ms=round(total_delay, 3),
                    bottleneck_bw_mbps=round(bottleneck_bw, 3),
                    notes="store_carry_forward",
                )
        return best_plan


ROUTER_REGISTRY = {
    "optimized": OptimizedRouter,
    "hypatia": HypatiaRouter,
    "lsr": LSRRouter,
    "madrl": MADRLRouter,
    "otcp": OTCPRouter,
    "ftrl": FTRLRouter,
    "dtn_cgr": DTNCGRRouter,
}


def create_router(router_name):
    router_class = ROUTER_REGISTRY.get(router_name)
    if router_class is None:
        raise ValueError(f"Unknown router: {router_name}")
    return router_class()


def build_context(time_ms, topology_hash, current_nodes_df, node_ip_map, active_links, node_loads, future_snapshots=None, flow_requests=None):
    graph_delay = build_graph_for_priority(active_links, "DELAY")
    graph_high = build_graph_for_priority(active_links, "HIGH")
    graph_normal = build_graph_for_priority(active_links, "NORMAL")
    node_positions = {
        row["node_id"]: np.array([row["ecef_x"], row["ecef_y"], row["ecef_z"]], dtype=float)
        for _, row in current_nodes_df.iterrows()
    }
    type_map = build_type_map(current_nodes_df)
    return RoutingContext(
        time_ms=time_ms,
        topology_hash=topology_hash,
        node_positions=node_positions,
        node_ip_map=node_ip_map,
        graph_delay=graph_delay,
        graph_high=graph_high,
        graph_normal=graph_normal,
        active_edge_lookup=make_edge_lookup(graph_delay),
        node_loads=node_loads,
        type_map=type_map,
        future_snapshots=future_snapshots,
        flow_requests=flow_requests,
    )


def build_future_snapshots(df_sat, df_uav, timelines, start_index, horizon):
    snapshots = []
    end_index = min(start_index + horizon + 1, len(timelines))
    for future_index in range(start_index, end_index):
        future_time_ms = int(timelines[future_index])
        future_nodes_df = get_nodes_at_timestamp(df_sat, df_uav, future_time_ms)
        if future_nodes_df.empty:
            continue
        future_links = compute_topology(future_nodes_df, future_time_ms)
        active_links = [link for link in future_links if link.get("status", "UP") == "UP"]
        future_type_map = build_type_map(future_nodes_df)
        future_graph = build_effective_delay_graph(active_links, future_nodes_df)
        cache_nodes = select_otcp_candidate_sats(future_graph, future_type_map)
        snapshots.append(
            {
                "G": future_graph,
                "cache_nodes": cache_nodes,
                "type_map": future_type_map,
            }
        )
    return snapshots


def save_chunk(output_link_dir, output_rule_dir, chunk_index, start_ms, end_ms, chunk_links, chunk_rules):
    link_filename = f"topology_links_{start_ms}_{end_ms}.csv"
    rule_filename = f"routing_rules_{start_ms}_{end_ms}.json"

    if chunk_links:
        df_links = pd.DataFrame(chunk_links)
        link_columns = [
            "time_ms",
            "src",
            "dst",
            "direction",
            "distance_km",
            "delay_ms",
            "jitter_ms",
            "loss_pct",
            "bw_mbps",
            "max_queue_pkt",
            "type",
            "status",
            "lifetime_ms",
        ]
        for column in link_columns:
            if column not in df_links.columns:
                df_links[column] = None
        df_links[link_columns].to_csv(os.path.join(output_link_dir, link_filename), index=False)

    payload = {"meta": {"chunk_id": chunk_index, "version": "standalone-v1"}, "rules": chunk_rules}
    with open(os.path.join(output_rule_dir, rule_filename), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyEncoder)


def get_satellite_scale_label(sat_dir):
    sat_name = os.path.basename(os.path.normpath(sat_dir))
    if sat_name == "sat_trace":
        return "25"
    if sat_name.startswith("sat_trace_"):
        suffix = sat_name.replace("sat_trace_", "", 1)
        if suffix.isdigit():
            return suffix
    return sat_name


def sanitize_output_tag(output_tag):
    if not output_tag:
        return None
    sanitized = output_tag.strip().replace(" ", "_")
    sanitized = sanitized.replace("/", "_").replace("\\", "_")
    return sanitized or None


def run_simulation(
    router_name,
    sat_dir="traces/sat_trace",
    uav_file="traces/uav_trace_full.csv",
    save_outputs=True,
    max_steps=None,
    output_tag=None,
    traffic_model=TRAFFIC_MODEL_LEGACY,
    photo_interval_s=DEFAULT_PHOTO_INTERVAL_S,
    photo_size_mb=DEFAULT_PHOTO_SIZE_MB,
    include_ctrl_flow=True,
):
    router = create_router(router_name)
    start_time = time.perf_counter()

    df_sat, df_uav, timelines = load_and_merge_traces(sat_dir=sat_dir, uav_file=uav_file)
    if not timelines:
        raise RuntimeError("No timelines found in UAV trace.")
    if df_sat.empty:
        raise RuntimeError(f"No satellite trace data found under {sat_dir}.")

    if max_steps is not None:
        timelines = timelines[:max_steps]

    node_ip_map = build_node_ip_map(df_sat, df_uav)
    scale_label = get_satellite_scale_label(sat_dir)
    resolved_output_tag = sanitize_output_tag(output_tag)
    output_suffix = resolved_output_tag or scale_label
    output_root = os.path.join("outputs", f"output_{output_suffix}")
    output_link_dir = os.path.join(output_root, "links")
    output_rule_dir = os.path.join(output_root, "rules")
    if save_outputs:
        os.makedirs(output_link_dir, exist_ok=True)
        os.makedirs(output_rule_dir, exist_ok=True)

    stats = SimulationStats(router.algorithm_name)
    topo_cache = TopologyCache()
    last_topology_step = -TOPO_HASH_INTERVAL
    chunk_size_ms = 60000
    chunk_links = []
    chunk_rules = []
    chunk_index = 0

    for step_index, timestamp in enumerate(timelines):
        time_ms = int(timestamp)
        nodes_df = get_nodes_at_timestamp(df_sat, df_uav, time_ms)
        topology_changed = False
        reused_topology = False
        if step_index - last_topology_step >= TOPO_HASH_INTERVAL or step_index == 0:
            topo_start = time.perf_counter()
            links = compute_topology(nodes_df, time_ms)
            stats.topology_compute_time_ms += (time.perf_counter() - topo_start) * 1000.0
            topology_changed = topo_cache.is_topology_changed(links)
            topo_cache.cache_topology(links)
            last_topology_step = step_index
        else:
            reused_topology = True
            links = []
            for link in topo_cache.last_topology or []:
                copied_link = link.copy()
                copied_link["time_ms"] = time_ms
                links.append(copied_link)

        node_ids = nodes_df["node_id"].values if not nodes_df.empty else []
        should_emit_s4_rules = traffic_model != TRAFFIC_MODEL_S4_COMPAT or topology_changed or step_index == 0
        if should_emit_s4_rules:
            flow_requests = build_flow_requests(
                node_ids,
                time_ms,
                traffic_model=traffic_model,
                photo_interval_s=photo_interval_s,
                photo_size_mb=photo_size_mb,
                include_ctrl_flow=include_ctrl_flow,
            )
        else:
            flow_requests = []

        export_flow_requests = flow_requests
        if traffic_model == TRAFFIC_MODEL_PERIODIC_PHOTO and not flow_requests:
            export_flow_requests = build_flow_requests(
                node_ids,
                time_ms,
                traffic_model=traffic_model,
                photo_interval_s=photo_interval_s,
                photo_size_mb=photo_size_mb,
                include_ctrl_flow=include_ctrl_flow,
                emit_photo_every_step=True,
            )

        active_links = [link for link in links if link.get("status", "UP") == "UP"]
        node_loads = defaultdict(int)
        future_snapshots = None
        if router.requires_future_snapshots:
            future_snapshots = build_future_snapshots(df_sat, df_uav, timelines, step_index, OTCP_LOOKAHEAD_HORIZON)
        context = build_context(
            time_ms,
            topo_cache.last_topology_hash,
            nodes_df,
            node_ip_map,
            active_links,
            node_loads,
            future_snapshots=future_snapshots,
            flow_requests=flow_requests,
        )
        router.begin_step(context)

        stats.total_flow_requests += len(flow_requests)
        step_rules = []
        counted_flow_keys = {
            (flow_request["flow_name"], flow_request["src"], flow_request["dst"])
            for flow_request in flow_requests
        }

        for flow_request in export_flow_requests:
            route_start = time.perf_counter()
            plan = router.plan_route(flow_request, context)
            stats.routing_compute_time_ms += (time.perf_counter() - route_start) * 1000.0
            flow_key = (flow_request["flow_name"], flow_request["src"], flow_request["dst"])
            count_in_stats = flow_key in counted_flow_keys

            if plan is None or len(plan.path) < 2 or not should_export_plan(flow_request, plan):
                if count_in_stats:
                    stats.record_failure()
                continue

            step_rules.extend(make_rules(flow_request, plan, time_ms, node_ip_map))
            for node_id in plan.path[1:]:
                node_loads[node_id] += 1
            if count_in_stats:
                stats.record_success(plan)

        chunk_links.extend(links)
        chunk_rules.extend(step_rules)
        stats.record_step(len(active_links), reused_topology, topology_changed)

        is_last_step = step_index == len(timelines) - 1
        next_timestamp = timelines[step_index + 1] if not is_last_step else -1
        if is_last_step or int(next_timestamp / chunk_size_ms) > int(time_ms / chunk_size_ms):
            if save_outputs:
                start_ms = chunk_index * chunk_size_ms
                save_chunk(output_link_dir, output_rule_dir, chunk_index, start_ms, time_ms, chunk_links, chunk_rules)
            chunk_links = []
            chunk_rules = []
            chunk_index += 1

    runtime_seconds = time.perf_counter() - start_time
    metrics = stats.to_dict(runtime_seconds, sat_dir, max_steps)
    metrics["UAV Trace"] = uav_file
    metrics["Traffic Model"] = traffic_model
    metrics["Photo Interval (s)"] = round(float(photo_interval_s), 3)
    metrics["Photo Size (MB)"] = round(float(photo_size_mb), 3)
    metrics["Include CTRL Flow"] = bool(include_ctrl_flow)
    metrics["Output Root"] = output_root

    if save_outputs:
        metrics_path = os.path.join(output_root, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, cls=NumpyEncoder)

    return metrics


def run_cli(router_name):
    parser = argparse.ArgumentParser(description=f"Run {router_name} routing on the S3 topology project.")
    parser.add_argument("sat_dir", nargs="?", default="traces/sat_trace")
    parser.add_argument("--uav-file", dest="uav_file", default="traces/uav_trace_full.csv")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--traffic-model", choices=[TRAFFIC_MODEL_LEGACY, TRAFFIC_MODEL_PERIODIC_PHOTO, TRAFFIC_MODEL_S4_COMPAT], default=TRAFFIC_MODEL_LEGACY)
    parser.add_argument("--photo-interval-s", type=float, default=DEFAULT_PHOTO_INTERVAL_S)
    parser.add_argument("--photo-size-mb", type=float, default=DEFAULT_PHOTO_SIZE_MB)
    parser.add_argument("--no-ctrl-flow", action="store_true")
    args = parser.parse_args()

    metrics = run_simulation(
        router_name=router_name,
        sat_dir=args.sat_dir,
        uav_file=args.uav_file,
        save_outputs=not args.no_save,
        max_steps=args.max_steps,
        output_tag=args.output_tag,
        traffic_model=args.traffic_model,
        photo_interval_s=args.photo_interval_s,
        photo_size_mb=args.photo_size_mb,
        include_ctrl_flow=not args.no_ctrl_flow,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_BASE = os.path.join(SCRIPT_DIR, "traces")
OUTPUTS_BASE = os.path.join(SCRIPT_DIR, "outputs")

MAX_LINK_RANGE = 5000 * 1000
MIN_ELEVATION_DEG = 10.0
SPEED_OF_LIGHT = 3e8
TOPO_HASH_INTERVAL = 100

FLOWS = {
    "CTRL_FLOW": {
        "src": "GS_01",
        "priority": "HIGH",
        "base_bw_mbps": 0.01,
    },
    "VIDEO_FLOW_UAV_01": {
        "src": "UAV_01",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 180,
        "burst_bw_mbps": 40.0,
    },
    "VIDEO_FLOW_UAV_05": {
        "src": "UAV_05",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 300,
        "burst_bw_mbps": 40.0,
    },
    "VIDEO_FLOW_UAV_10": {
        "src": "UAV_10",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 360,
        "burst_bw_mbps": 40.0,
    },
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
            for link in sorted(links, key=lambda x: (x["src"], x["dst"]))
            if link.get("status", "UP") == "UP"
        )
        return hashlib.md5(link_str.encode("utf-8")).hexdigest()

    def is_topology_changed(self, links):
        new_hash = self.get_hash(links)
        changed = new_hash != self.last_topology_hash
        if changed:
            self.last_topology_hash = new_hash
        return changed


def scan_trace_dirs():
    """掃描 traces/ 下可用的 sat_trace_n 與 uav_trace_n 子目錄"""
    sat_base = os.path.join(TRACES_BASE, "sat_trace")
    uav_base = os.path.join(TRACES_BASE, "uav_trace")

    sat_dirs = {}
    if os.path.isdir(sat_base):
        for d in os.listdir(sat_base):
            full = os.path.join(sat_base, d)
            if d.startswith("sat_trace_") and os.path.isdir(full):
                suffix = d.replace("sat_trace_", "", 1)
                if suffix.isdigit():
                    sat_dirs[int(suffix)] = full

    uav_dirs = {}
    if os.path.isdir(uav_base):
        for d in os.listdir(uav_base):
            full = os.path.join(uav_base, d)
            if d.startswith("uav_trace_") and os.path.isdir(full):
                suffix = d.replace("uav_trace_", "", 1)
                if suffix.isdigit():
                    uav_dirs[int(suffix)] = full

    return sat_dirs, uav_dirs


def get_satellite_scale_label(sat_dir):
    sat_name = os.path.basename(os.path.normpath(sat_dir))
    if sat_name == "sat_trace":
        return "25"
    if sat_name.startswith("sat_trace_"):
        suffix = sat_name.replace("sat_trace_", "", 1)
        if suffix.isdigit():
            return suffix
    return sat_name


def load_and_merge_traces(sat_dir, uav_dir):
    """載入 sat_trace_n/ 與 uav_trace_n/ 目錄下的所有 CSV 並合併"""
    sat_files = sorted(glob.glob(os.path.join(sat_dir, "*.csv")))
    sat_frames = [pd.read_csv(f) for f in sat_files]
    df_sat = pd.concat(sat_frames, ignore_index=True) if sat_frames else pd.DataFrame()

    uav_files = sorted(glob.glob(os.path.join(uav_dir, "*.csv")))
    uav_frames = [pd.read_csv(f) for f in uav_files]
    df_uav = pd.concat(uav_frames, ignore_index=True) if uav_frames else pd.DataFrame()

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

    for source_idx in range(len(node_ids)):
        for neighbor_slot, target_idx in enumerate(neighbor_indices[source_idx]):
            if distances[source_idx][neighbor_slot] == float("inf") or source_idx == target_idx:
                continue

            src = node_ids[source_idx]
            dst = node_ids[target_idx]
            pair = (src, dst) if src < dst else (dst, src)
            if pair in processed_pairs:
                continue

            type_a = node_types[source_idx]
            type_b = node_types[target_idx]

            is_sat_a = type_a == "SAT"
            is_sat_b = type_b == "SAT"
            if is_sat_a != is_sat_b:
                sat_idx = source_idx if is_sat_a else target_idx
                ground_idx = target_idx if is_sat_a else source_idx
                elevation = calculate_elevation(coords[ground_idx], coords[sat_idx])
                if elevation < MIN_ELEVATION_DEG:
                    continue

            dist_m = distances[source_idx][neighbor_slot]
            delay_ms = calculate_delay(dist_m)
            bw_mbps = calculate_bandwidth(type_a, type_b)
            if bw_mbps == 0:
                continue

            links.append(
                {
                    "time_ms": time_ms,
                    "src": src,
                    "dst": dst,
                    "direction": "BIDIR",
                    "distance_km": round(dist_m / 1000.0, 3),
                    "delay_ms": round(delay_ms, 3),
                    "jitter_ms": round(calculate_jitter(delay_ms), 3),
                    "loss_pct": 0.0,
                    "bw_mbps": bw_mbps,
                    "max_queue_pkt": calculate_bdp_queue(bw_mbps, delay_ms),
                    "type": f"{type_a}-{type_b}",
                    "status": "UP",
                }
            )
            processed_pairs.add(pair)

    return links


def build_graph_for_priority(links, priority):
    graph = nx.Graph()
    for link in links:
        delay = link["delay_ms"]
        if priority == "HIGH":
            weight = delay * 0.1 + 16.7
        else:
            weight = delay + 0.83
        graph.add_edge(link["src"], link["dst"], weight=weight)
    return graph


def get_current_bandwidth(flow_name, flow_config, time_ms):
    time_s = time_ms / 1000.0
    if "burst_time_s" in flow_config and time_s >= flow_config["burst_time_s"]:
        base_bw = flow_config["burst_bw_mbps"]
    else:
        base_bw = flow_config["base_bw_mbps"]
    rng = random.Random(hash((flow_name, int(time_s * 10))))
    fluctuation = base_bw * rng.uniform(-0.05, 0.05)
    return round(max(0.01, base_bw + fluctuation), 2)


def build_flow_requests(active_nodes):
    requests = []
    node_set = set(active_nodes)

    if "GS_01" in node_set:
        # 所有在线 UAV 都有控制流 + 视频流
        target_uavs = sorted([n for n in node_set if n.startswith("UAV_")])

        for uav in target_uavs:
            requests.append((f"CTRL_FLOW_{uav}", "GS_01", uav, FLOWS["CTRL_FLOW"]))

        import copy
        for i, uav in enumerate(target_uavs):
            flow_name = f"VIDEO_FLOW_{uav}"
            # 兼容如果已经配了的就用原来的，没有的话顺延自动生成
            if flow_name in FLOWS:
                requests.append((flow_name, uav, "GS_01", FLOWS[flow_name]))
            else:
                cfg = {
                    "src": uav,
                    "priority": "NORMAL",
                    "base_bw_mbps": 10.0,
                    "burst_time_s": 180 + i * 60,
                    "burst_bw_mbps": 40.0,
                }
                requests.append((flow_name, uav, "GS_01", cfg))

        # 设定一个具体的 UAV -> UAV 通讯场景 (例如 UAV_01 到 UAV_05)
        if "UAV_01" in node_set and "UAV_05" in node_set:
            inter_uav_cfg = {
                "src": "UAV_01",
                "priority": "HIGH",
                "base_bw_mbps": 2.0,
            }
            requests.append(("INTER_UAV_FLOW_UAV_01_UAV_05", "UAV_01", "UAV_05", inter_uav_cfg))

    return requests


def build_node_ip_map(df_sat, df_uav):
    node_ip_map = {}
    for df in [df_sat, df_uav]:
        if df.empty:
            continue
        for _, row in df[["node_id", "ip"]].drop_duplicates().iterrows():
            node_ip_map[row["node_id"]] = row["ip"]
    return node_ip_map


def generate_routing_rules(active_links, time_ms, active_nodes, node_ip_map):
    rules = []
    if not active_links:
        return rules

    graph_high = build_graph_for_priority(active_links, "HIGH")
    graph_normal = build_graph_for_priority(active_links, "NORMAL")

    for flow_name, src, dst, flow_cfg in build_flow_requests(active_nodes):
        graph = graph_high if flow_cfg["priority"] == "HIGH" else graph_normal
        if not graph.has_node(src) or not graph.has_node(dst):
            continue
        try:
            path = nx.shortest_path(graph, src, dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2:
            continue

        for i in range(len(path) - 1):
            current_node = path[i]
            next_hop = path[i + 1]
            rules.append(
                {
                    "time_ms": int(time_ms),
                    "node": current_node,
                    "dst_cidr": f"{node_ip_map.get(dst, '0.0.0.0')}/32",
                    "action": "replace",
                    "next_hop": next_hop,
                    "next_hop_ip": node_ip_map.get(next_hop, "0.0.0.0"),
                    "algo": "Optimized",
                    "req_bw_mbps": get_current_bandwidth(flow_name, flow_cfg, time_ms),
                    "debug_info": f"{src}->{dst}",
                }
            )

    return rules


def save_chunk(output_link_dir, output_rule_dir, chunk_idx, start_ms, end_ms, chunk_links, chunk_rules):
    link_filename = f"topology_links_{start_ms}_{end_ms}.csv"
    rule_filename = f"routing_rules_{start_ms}_{end_ms}.json"

    if chunk_links:
        df_links = pd.DataFrame(chunk_links)
        columns = [
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
        ]
        for col in columns:
            if col not in df_links.columns:
                df_links[col] = None
        df_links[columns].to_csv(os.path.join(output_link_dir, link_filename), index=False)

    payload = {"meta": {"chunk_id": chunk_idx, "version": "standalone-v1"}, "rules": chunk_rules}
    with open(os.path.join(output_rule_dir, rule_filename), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="Standalone optimized S3 routing script")
    parser.add_argument(“--sat-dir”, default=None, help=”直接指定 sat_trace 子目錄路徑（跳過互動選擇）”)
    parser.add_argument(“--uav-dir”, default=None, help=”直接指定 uav_trace 子目錄路徑（跳過互動選擇）”)
    parser.add_argument(“--max-steps”, type=int, default=None)
    parser.add_argument(“--no-save”, action=”store_true”)
    args = parser.parse_args()

    # ── 1. 掃描可用資料夾 ──
    sat_dirs, uav_dirs = scan_trace_dirs()

    if args.sat_dir:
        sat_path = args.sat_dir
        sat_n = get_satellite_scale_label(sat_path)
    else:
        if not sat_dirs:
            print(“[Error] traces/sat_trace/ 下沒有任何 sat_trace_n 目錄，請先執行 S1”)
            return
        print(“\n可用的 sat_trace_n 目錄：”)
        for n in sorted(sat_dirs):
            print(f”  sat_trace_{n}”)
        while True:
            try:
                choice = input(“請輸入要使用的 sat n (數量): “).strip()
                sat_n = int(choice)
                if sat_n in sat_dirs:
                    sat_path = sat_dirs[sat_n]
                    break
                print(f”  無效選擇，可用 n: {sorted(sat_dirs)}”)
            except (ValueError, EOFError, KeyboardInterrupt):
                print(“\n[Exit]”)
                return

    if args.uav_dir:
        uav_path = args.uav_dir
        uav_n = get_satellite_scale_label(uav_path)
    else:
        if not uav_dirs:
            print(“[Error] traces/uav_trace/ 下沒有任何 uav_trace_n 目錄，請先執行 S2”)
            return
        print(“\n可用的 uav_trace_n 目錄：”)
        for n in sorted(uav_dirs):
            print(f”  uav_trace_{n}”)
        while True:
            try:
                choice = input(“請輸入要使用的 uav n (數量): “).strip()
                uav_n = int(choice)
                if uav_n in uav_dirs:
                    uav_path = uav_dirs[uav_n]
                    break
                print(f”  無效選擇，可用 n: {sorted(uav_dirs)}”)
            except (ValueError, EOFError, KeyboardInterrupt):
                print(“\n[Exit]”)
                return

    print(f”\n🛰️  使用 sat : {sat_path}”)
    print(f”🚁 使用 uav : {uav_path}”)

    # ── 2. 載入資料 ──
    df_sat, df_uav, timelines = load_and_merge_traces(sat_path, uav_path)
    if df_sat.empty:
        print(f”[Error] 無衛星資料：{sat_path}”)
        return
    if not timelines:
        print(f”[Error] 無 UAV 時間軸：{uav_path}”)
        return

    if args.max_steps is not None:
        timelines = timelines[: args.max_steps]

    # ── 3. 輸出目錄 output_m_n ──
    output_root = os.path.join(OUTPUTS_BASE, f”output_{sat_n}_{uav_n}”)
    output_link_dir = os.path.join(output_root, “links”)
    output_rule_dir = os.path.join(output_root, “rules”)
    if not args.no_save:
        if os.path.exists(output_root):
            shutil.rmtree(output_root)
        os.makedirs(output_link_dir, exist_ok=True)
        os.makedirs(output_rule_dir, exist_ok=True)
    print(f”📁 輸出目錄：{output_root}”)

    # ── 4. 計算拓撲與路由 ──
    node_ip_map = build_node_ip_map(df_sat, df_uav)
    topo_cache = TopologyCache()
    chunk_size_ms = 60000
    chunk_idx = 0
    chunk_links = []
    chunk_rules = []
    last_topo_step = -TOPO_HASH_INTERVAL

    for step_idx, t in enumerate(timelines):
        time_ms = int(t)
        nodes_df = get_nodes_at_timestamp(df_sat, df_uav, time_ms)
        active_nodes = nodes_df[“node_id”].values if not nodes_df.empty else []

        if step_idx - last_topo_step >= TOPO_HASH_INTERVAL or step_idx == 0:
            links = compute_topology(nodes_df, time_ms)

            if topo_cache.last_topology:
                current_keys = {
                    (l[“src”], l[“dst”]) if l[“src”] < l[“dst”] else (l[“dst”], l[“src”])
                    for l in links
                    if l.get(“status”, “UP”) == “UP”
                }
                for old_link in topo_cache.last_topology:
                    if old_link.get(“status”, “UP”) != “UP”:
                        continue
                    old_key = (
                        (old_link[“src”], old_link[“dst”])
                        if old_link[“src”] < old_link[“dst”]
                        else (old_link[“dst”], old_link[“src”])
                    )
                    if old_key not in current_keys:
                        broken = old_link.copy()
                        broken[“time_ms”] = time_ms
                        broken[“status”] = “DOWN”
                        broken[“delay_ms”] = 99999.0
                        links.append(broken)

            topo_cache.last_topology = links
            topo_cache.is_topology_changed(links)
            last_topo_step = step_idx
        else:
            links = []
            for old_link in topo_cache.last_topology or []:
                copied = old_link.copy()
                copied[“time_ms”] = time_ms
                links.append(copied)

        active_links = [link for link in links if link.get(“status”, “UP”) == “UP”]
        new_rules = generate_routing_rules(active_links, time_ms, active_nodes, node_ip_map)

        chunk_links.extend(links)
        chunk_rules.extend(new_rules)

        is_last_step = step_idx == len(timelines) - 1
        next_t = timelines[step_idx + 1] if not is_last_step else -1
        if is_last_step or int(next_t / chunk_size_ms) > int(time_ms / chunk_size_ms):
            if not args.no_save:
                start_ms = chunk_idx * chunk_size_ms
                save_chunk(output_link_dir, output_rule_dir, chunk_idx, start_ms, time_ms, chunk_links, chunk_rules)
            chunk_links = []
            chunk_rules = []
            chunk_idx += 1

    print(f”[Done] output_{sat_n}_{uav_n} 完成。”)


if __name__ == “__main__”:
    main()


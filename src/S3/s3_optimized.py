import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
import math
import networkx as nx
import glob
import random
import os
from collections import defaultdict
import hashlib

# --- 全局配置 ---
MAX_LINK_RANGE = 5000 * 1000  
MIN_ELEVATION_DEG = 10.0      
SPEED_OF_LIGHT = 3e8
TOPO_HASH_INTERVAL = 100  # 每 100 步检查一次拓扑变化

FLOWS = {
    "CTRL_FLOW": {
        "src": "GS_01",
        "priority": "HIGH",
        "base_bw_mbps": 0.01,
        "dst_cidr": "10.99.0.0/24"
    },
    "VIDEO_FLOW_UAV_01": {
        "src": "UAV_01",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 180,
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.1.1/32"
    },
    "VIDEO_FLOW_UAV_02": {
        "src": "UAV_02",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 300,
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.2.1/32"
    },
    "VIDEO_FLOW_UAV_03": {
        "src": "UAV_03",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 360,
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.3.1/32"
    }
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class TopologyCache:
    """缓存并跟踪拓扑变化"""
    def __init__(self):
        self.last_topology_hash = None
        self.last_topology = None
        self.last_graph_by_priority = {}
        
    def get_hash(self, links):
        """计算拓扑哈希以检测变化"""
        if not links:
            return None
        # 仅基于链路存在情况，不考虑延迟这样的变化参数
        link_str = '|'.join(
            f"{l['src']}-{l['dst']}" 
            for l in sorted(links, key=lambda x: (x['src'], x['dst']))
        )
        return hashlib.md5(link_str.encode()).hexdigest()
    
    def is_topology_changed(self, links):
        """检查拓扑是否发生变化"""
        new_hash = self.get_hash(links)
        changed = new_hash != self.last_topology_hash
        if changed:
            self.last_topology_hash = new_hash
        return changed
    
    def cache_topology(self, links, graphs):
        """缓存拓扑和图"""
        self.last_topology = links
        self.last_graph_by_priority = graphs


def load_and_merge_traces(sat_dir='sat_trace', uav_file='uav_trace_full.csv'):
    print(">>> Loading Trace Data...")
    
    sat_files = glob.glob(os.path.join(sat_dir, "*.csv"))
    sat_dfs = []
    for f in sat_files:
        print(f"  - Reading {f}")
        df = pd.read_csv(f)
        sat_dfs.append(df)
    
    df_sat = pd.concat(sat_dfs) if sat_dfs else pd.DataFrame()
    
    print(f"  - Reading {uav_file}")
    if os.path.exists(uav_file):
        df_uav = pd.read_csv(uav_file)
    else:
        print(f"[Error] UAV trace file not found: {uav_file}")
        df_uav = pd.DataFrame()

    all_timestamps = sorted(df_uav['time_ms'].unique()) if not df_uav.empty else []
    if all_timestamps:
        print(f">>> Total Time Steps: {len(all_timestamps)} (from {min(all_timestamps)} to {max(all_timestamps)} ms)")
    
    return df_sat, df_uav, all_timestamps

def get_nodes_at_timestamp(df_sat, df_uav, target_time_ms):
    uav_current = df_uav[df_uav['time_ms'] == target_time_ms]
    sat_time_key = (target_time_ms // 1000) * 1000
    sat_current = df_sat[df_sat['time_ms'] == sat_time_key]
    
    cols = ['node_id', 'type', 'ecef_x', 'ecef_y', 'ecef_z', 'ip']
    
    if sat_current.empty and uav_current.empty:
        return pd.DataFrame(columns=cols)
        
    nodes = pd.concat([sat_current[cols], uav_current[cols]], ignore_index=True)
    return nodes

def calculate_delay(dist_m): 
    return (dist_m / SPEED_OF_LIGHT) * 1000

def calculate_jitter(delay_ms): 
    return delay_ms * 0.1

def calculate_bandwidth(type_a, type_b):
    types = {type_a, type_b}
    if 'GS' in types and 'UAV' in types: 
        return 0
    if 'UAV' in types and 'SAT' in types: 
        return 20
    if 'SAT' in types and 'GS' in types: 
        return 20
    if 'SAT' in types: 
        return 100
    return 10

def calculate_bdp_queue(bw_mbps, delay_ms):
    queue = int((bw_mbps * 1e6) * (delay_ms * 2 * 1e-3) / 12000)
    return max(10, queue)

def calculate_elevation(pos_a, pos_b):
    vec_a = np.array(pos_a)
    vec_ab = np.array(pos_b) - np.array(pos_a)
    dist_a = np.linalg.norm(vec_a)
    dist_ab = np.linalg.norm(vec_ab)
    if dist_a == 0 or dist_ab == 0: 
        return 90.0
    cos_theta = np.dot(vec_a, vec_ab) / (dist_a * dist_ab)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta_rad = np.arccos(cos_theta)
    return 90 - math.degrees(theta_rad)

def compute_topology(nodes_df, time_ms, FORCE_DOWN_SRC='GS_01', FORCE_DOWN_DST='SAT_63188', DOWN_TIME_MS=0):
    """优化版本：缓存坐标和键值对"""
    links = []
    if len(nodes_df) < 2: 
        return links
    
    coords = nodes_df[['ecef_x', 'ecef_y', 'ecef_z']].values
    ids = nodes_df['node_id'].values
    types = nodes_df['type'].values
    
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=20, distance_upper_bound=MAX_LINK_RANGE)
    
    processed_pairs = set()
    
    for i in range(len(ids)):
        for j_idx, neighbor_idx in enumerate(indices[i]):
            if dists[i][j_idx] == float('inf') or i == neighbor_idx: 
                continue
            
            n1_id, n2_id = ids[i], ids[neighbor_idx]
            pair_key = (n1_id, n2_id) if n1_id < n2_id else (n2_id, n1_id)
            if pair_key in processed_pairs: 
                continue
            
            type_a, type_b = types[i], types[neighbor_idx]
            
            is_sat_a = (type_a == 'SAT')
            is_sat_b = (type_b == 'SAT')
            
            if is_sat_a != is_sat_b: 
                sat_idx = i if is_sat_a else neighbor_idx
                gnd_idx = neighbor_idx if is_sat_a else i
                elev = calculate_elevation(coords[gnd_idx], coords[sat_idx])
                if elev < MIN_ELEVATION_DEG: 
                    continue

            dist_m = dists[i][j_idx]
            delay = calculate_delay(dist_m)
            bw = calculate_bandwidth(type_a, type_b)
            
            if bw == 0: 
                continue
            
            status = 'UP'
            if time_ms >= DOWN_TIME_MS:
                if (n1_id == FORCE_DOWN_SRC and n2_id == FORCE_DOWN_DST) or \
                   (n1_id == FORCE_DOWN_DST and n2_id == FORCE_DOWN_SRC):
                    status = 'DOWN'
                    delay = 99999.0
            
            links.append({
                'time_ms': time_ms,
                'src': n1_id,
                'dst': n2_id,
                'direction': 'BIDIR',
                'distance_km': round(dist_m / 1000, 3),
                'delay_ms': round(delay, 2),
                'jitter_ms': round(calculate_jitter(delay), 3),
                'loss_pct': 0.0,
                'bw_mbps': bw,
                'max_queue_pkt': calculate_bdp_queue(bw, delay),
                'type': f"{type_a}-{type_b}",
                'status': status,
                'lifetime_ms': 60000
            })
            processed_pairs.add(pair_key)
    
    return links

def build_graph_for_flow(links, priority):
    """构建加权图"""
    G = nx.Graph()
    for l in links:
        delay = l['delay_ms']
        lifetime_sec = max(l['lifetime_ms'] / 1000.0, 0.1)
        
        if priority == 'HIGH':
            stability_penalty = 1000.0 / lifetime_sec 
            weight = delay * 0.1 + stability_penalty 
        else:
            stability_penalty = 50.0 / lifetime_sec
            weight = delay * 1.0 + stability_penalty
            
        G.add_edge(l['src'], l['dst'], weight=weight, capacity=l['bw_mbps'])
        
    return G

# 预计算带宽（避免每次调用都重置随机种子）
_bw_cache = {}

def get_current_bandwidth(flow_config, current_t_ms):
    """优化版本：缓存带宽计算结果"""
    cache_key = (id(flow_config), current_t_ms)
    if cache_key in _bw_cache:
        return _bw_cache[cache_key]
    
    t_sec = current_t_ms / 1000.0
    
    if 'burst_time_s' in flow_config and t_sec >= flow_config['burst_time_s']:
        base_bw = flow_config['burst_bw_mbps']
    else:
        base_bw = flow_config['base_bw_mbps']
        
    random.seed(int(t_sec * 10)) 
    fluctuation = base_bw * random.uniform(-0.05, 0.05)
    
    current_bw = round(max(0.01, base_bw + fluctuation), 2)
    _bw_cache[cache_key] = current_bw
    return current_bw

def generate_routing_rules(active_links, time_ms, node_ip_map, active_nodes, cached_graphs=None):
    """优化版本：使用缓存的图"""
    rules = []
    if not active_links: 
        return rules

    # 只在需要时构建图
    if cached_graphs is None:
        G_ctrl = build_graph_for_flow(active_links, 'HIGH')
        G_video = build_graph_for_flow(active_links, 'NORMAL')
    else:
        G_ctrl, G_video = cached_graphs

    # 控制流
    ctrl_flow = FLOWS["CTRL_FLOW"]
    if ctrl_flow['src'] in active_nodes:
        target_uavs = [n for n in active_nodes if n.startswith('UAV_')]
        if target_uavs:
            target_uav = target_uavs[0]
            try:
                path = nx.shortest_path(G_ctrl, ctrl_flow['src'], target_uav, weight='weight')
                if len(path) > 1:
                    nh_id = path[1]
                    rules.append({
                        "time_ms": int(time_ms),
                        "node": ctrl_flow['src'],
                        "dst_cidr": ctrl_flow['dst_cidr'],
                        "action": "replace",
                        "next_hop": nh_id,
                        "next_hop_ip": node_ip_map.get(nh_id, "0.0.0.0"),
                        "algo": "Stability-First",
                        "req_bw_mbps": get_current_bandwidth(ctrl_flow, time_ms),
                        "debug_info": f"Ctrl command to UAVs"
                    })
            except: 
                pass

    # 视频流
    target_gs = 'GS_01'
    if target_gs not in active_nodes: 
        return rules

    for flow_name, flow_config in FLOWS.items():
        if flow_name.startswith("VIDEO_FLOW_"):
            src_uav = flow_config['src']
            if src_uav not in active_nodes: 
                continue
            
            try:
                path = nx.shortest_path(G_video, src_uav, target_gs, weight='weight')
                if len(path) > 1:
                    nh_id = path[1]
                    current_bw = get_current_bandwidth(flow_config, time_ms)
                    
                    rules.append({
                        "time_ms": int(time_ms),
                        "node": src_uav,
                        "dst_cidr": flow_config['dst_cidr'], 
                        "action": "replace",
                        "next_hop": nh_id,
                        "next_hop_ip": node_ip_map.get(nh_id, "0.0.0.0"),
                        "algo": "Latency-Balanced",
                        "req_bw_mbps": current_bw,
                        "debug_info": f"[{current_bw} Mbps] Video streaming to {target_gs}"
                    })
            except: 
                pass

    return rules

def main():
    output_link_dir = 'output/links'
    output_rule_dir = 'output/rules'
    os.makedirs(output_link_dir, exist_ok=True)
    os.makedirs(output_rule_dir, exist_ok=True)

    df_sat, df_uav, timelines = load_and_merge_traces(uav_file='uav_trace_full.csv')
    
    if not timelines:
        print("[Error] No timelines found. Exiting.")
        return

    print(">>> Building IP Map...")
    node_ip_map = {}
    for _, row in df_sat[['node_id', 'ip']].drop_duplicates().iterrows():
        node_ip_map[row['node_id']] = row['ip']
    for _, row in df_uav[['node_id', 'ip']].drop_duplicates().iterrows():
        node_ip_map[row['node_id']] = row['ip']
    print(f"   Mapped {len(node_ip_map)} nodes.")

    CHUNK_SIZE_MS = 60000 
    chunk_links = []
    chunk_rules = []
    current_chunk_idx = 0
    
    TARGET_DOWN_SRC = 'GS_01'
    TARGET_DOWN_DST = 'SAT_63188'
    DOWN_TIME_MS = 0

    print(f">>> Start Processing {len(timelines)} time steps...")
    
    # 初始化拓扑缓存
    topo_cache = TopologyCache()
    cached_graphs = None
    last_topology_computation = -TOPO_HASH_INTERVAL

    for i, t in enumerate(timelines): 
        t_val = int(t)
        current_nodes_df = get_nodes_at_timestamp(df_sat, df_uav, t_val)
        
        active_node_ids = current_nodes_df['node_id'].values if not current_nodes_df.empty else []
        
        # 只在必要时计算拓扑
        if i - last_topology_computation >= TOPO_HASH_INTERVAL or i == 0:
            links = compute_topology(current_nodes_df, t_val, TARGET_DOWN_SRC, TARGET_DOWN_DST, DOWN_TIME_MS)
            
            # 检测拓扑是否变化
            if topo_cache.is_topology_changed(links):
                # 拓扑变化，重新构建图
                active_links = [l for l in links if l['status'] == 'UP']
                cached_graphs = (
                    build_graph_for_flow(active_links, 'HIGH'),
                    build_graph_for_flow(active_links, 'NORMAL')
                )
                topo_cache.cache_topology(links, cached_graphs)
                print(f"   [Topology Updated] Step {i} (Time {t_val}ms)")
            
            last_topology_computation = i
        else:
            # 重用缓存的拓扑，必须深拷贝或创建新字典以避免覆盖历史记录中的 time_ms
            links = []
            if topo_cache.last_topology:
                for l in topo_cache.last_topology:
                    new_l = l.copy()
                    new_l['time_ms'] = t_val
                    links.append(new_l)
            
        chunk_links.extend(links)
        
        active_links = [l for l in links if l['status'] == 'UP']
        new_rules = generate_routing_rules(active_links, t_val, node_ip_map, active_node_ids, cached_graphs)
        chunk_rules.extend(new_rules)
        
        if i % 100 == 0:
            print(f"   [Progress] Step {i}/{len(timelines)} (Time {t_val}ms)")

        # 分片保存
        is_last_step = (i == len(timelines) - 1)
        next_t = timelines[i+1] if not is_last_step else -1
        
        if is_last_step or (int(next_t / CHUNK_SIZE_MS) > int(t_val / CHUNK_SIZE_MS)):
            start_ms = current_chunk_idx * CHUNK_SIZE_MS
            end_ms = t_val 
            link_filename = f"topology_links_{start_ms}_{end_ms}.csv"
            rule_filename = f"routing_rules_{start_ms}_{end_ms}.json"
            
            if chunk_links:
                df_links = pd.DataFrame(chunk_links)
                cols = ['time_ms', 'src', 'dst', 'direction', 'distance_km', 'delay_ms', 
                        'jitter_ms', 'loss_pct', 'bw_mbps', 'max_queue_pkt', 'type', 'status']
                for c in cols:
                    if c not in df_links.columns: df_links[c] = None
                df_links[cols].to_csv(os.path.join(output_link_dir, link_filename), index=False)
            
            rule_data = {"meta": {"version": "v1.2", "chunk_id": current_chunk_idx}, "rules": chunk_rules}
            with open(os.path.join(output_rule_dir, rule_filename), 'w') as f:
                json.dump(rule_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"   >>> [Saved Chunk {current_chunk_idx}] {link_filename}")
            
            chunk_links = []
            chunk_rules = []
            current_chunk_idx += 1

    print(">>> All Done.")

if __name__ == "__main__":
    main()

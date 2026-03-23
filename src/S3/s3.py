import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
import math
import networkx as nx
import glob
import random
import os

# --- 全局配置 ---
MAX_LINK_RANGE = 5000 * 1000  
MIN_ELEVATION_DEG = 10.0      
SPEED_OF_LIGHT = 3e8             

# 这里我们不再使用 CONTENT_LOCATIONS 找缓存，而是建立持续的端到端数据流
FLOWS = {
    # 1. 全局控制流 (生命线)：GS -> 所有 UAV (双向也行，这里模拟下行指令)
    "CTRL_FLOW": {
        "src": "GS_01",
        "priority": "HIGH",     # 高优：只看重稳定性 (Lifetime)
        "base_bw_mbps": 0.01,   # 10 kbps = 0.01 Mbps
        "dst_cidr": "10.99.0.0/24" # 虚拟指令网段
    },
    
    # 2. 视频回传流 (UAV -> GS_01)
    "VIDEO_FLOW_UAV_01": {
        "src": "UAV_01",
        "priority": "NORMAL",   # 普通：看重延迟和带宽
        "base_bw_mbps": 10.0,   # 初始低清 10 Mbps
        "burst_time_s": 180,    # 第 3 分钟爆发
        "burst_bw_mbps": 40.0,  # 爆发后高清 40 Mbps
        "dst_cidr": "10.88.1.1/32" # 回传目标虚拟 IP
    },
    "VIDEO_FLOW_UAV_02": {
        "src": "UAV_02",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 300,    # 第 5 分钟爆发
        "burst_bw_mbps": 40.0,
        "dst_cidr": "10.88.2.1/32"
    },
    "VIDEO_FLOW_UAV_03": {
        "src": "UAV_03",
        "priority": "NORMAL",
        "base_bw_mbps": 10.0,
        "burst_time_s": 360,    # 第 6 分钟爆发
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

# ---------------------------------------------------------
# 模块 0: 数据加载与融合 (修改点 1: 读取单个 UAV 文件)
# ---------------------------------------------------------
def load_and_merge_traces(sat_dir='sat_trace', uav_file='uav_trace_full.csv'):
    print(">>> Loading Trace Data...")
    
    # 1. 读取所有卫星数据 (保持不变)
    sat_files = glob.glob(os.path.join(sat_dir, "*.csv"))
    sat_dfs = []
    for f in sat_files:
        print(f"  - Reading {f}")
        df = pd.read_csv(f)
        sat_dfs.append(df)
    
    df_sat = pd.concat(sat_dfs) if sat_dfs else pd.DataFrame()
    
    # 2. 读取单个 UAV/地面站数据 (修改为直接读取文件)
    print(f"  - Reading {uav_file}")
    if os.path.exists(uav_file):
        df_uav = pd.read_csv(uav_file)
    else:
        print(f"[Error] UAV trace file not found: {uav_file}")
        df_uav = pd.DataFrame()

    # 3. 时间对齐
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

# ---------------------------------------------------------
# 模块 1 & 2: 物理层计算 (保持不变)
# ---------------------------------------------------------
def calculate_delay(dist_m): return (dist_m / SPEED_OF_LIGHT) * 1000
def calculate_jitter(delay_ms): return delay_ms * 0.1
def calculate_bandwidth(type_a, type_b):
    types = {type_a, type_b}
    if 'GS' in types and 'UAV' in types: 
        return 0  # 地面站和无人机之间距离太长，必须通过卫星中继
    if 'UAV' in types and 'SAT' in types: return 20
    if 'SAT' in types and 'GS' in types: return 20
    if 'SAT' in types: return 100
    return 10
def calculate_bdp_queue(bw_mbps, delay_ms):
    queue = int((bw_mbps * 1e6) * (delay_ms * 2 * 1e-3) / 12000)
    return max(10, queue)

def calculate_elevation(pos_a, pos_b):
    vec_a = np.array(pos_a)
    vec_ab = np.array(pos_b) - np.array(pos_a)
    dist_a = np.linalg.norm(vec_a)
    dist_ab = np.linalg.norm(vec_ab)
    if dist_a == 0 or dist_ab == 0: return 90.0
    cos_theta = np.dot(vec_a, vec_ab) / (dist_a * dist_ab)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta_rad = np.arccos(cos_theta)
    return 90 - math.degrees(theta_rad)

def compute_topology(nodes_df, time_ms):
    links = []
    if len(nodes_df) < 2: return links
    
    coords = nodes_df[['ecef_x', 'ecef_y', 'ecef_z']].values
    ids = nodes_df['node_id'].values
    types = nodes_df['type'].values
    
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=20, distance_upper_bound=MAX_LINK_RANGE)
    
    processed_pairs = set()

    # ★★★ 强制断连配置 ★★★
    # 确保这里的 ID 和你 CSV 里的一模一样
    # 如果不确定，可以写 print(ids) 看一下到底有哪些 ID
    FORCE_DOWN_SRC = 'GS_01'
    FORCE_DOWN_DST = 'SAT_63188'
    DOWN_TIME_MS = 0 # 0ms 开始断

    for i in range(len(ids)):
        for j_idx, neighbor_idx in enumerate(indices[i]):
            if dists[i][j_idx] == float('inf') or i == neighbor_idx: continue
            
            n1_id, n2_id = ids[i], ids[neighbor_idx]
            pair_key = tuple(sorted([n1_id, n2_id]))
            if pair_key in processed_pairs: continue
            
            type_a, type_b = types[i], types[neighbor_idx]
            
            is_sat_a = (type_a == 'SAT')
            is_sat_b = (type_b == 'SAT')
            
            if is_sat_a != is_sat_b: 
                sat_idx = i if is_sat_a else neighbor_idx
                gnd_idx = neighbor_idx if is_sat_a else i
                elev = calculate_elevation(coords[gnd_idx], coords[sat_idx])
                if elev < MIN_ELEVATION_DEG: continue

            dist_m = dists[i][j_idx]
            delay = calculate_delay(dist_m)
            bw = calculate_bandwidth(type_a, type_b)
            
            if bw == 0: continue  # 跳过带宽为 0 的链接
            
            # ★★★ 注入断连逻辑 ★★★
            status = 'UP'
            # 检查是否命中了我们要断开的链路，并且时间已经到了
            if time_ms >= DOWN_TIME_MS:
                if (n1_id == FORCE_DOWN_SRC and n2_id == FORCE_DOWN_DST) or \
                   (n1_id == FORCE_DOWN_DST and n2_id == FORCE_DOWN_SRC):
                    status = 'DOWN'
                    delay = 99999.0 # 极大延迟
                    # 可选：如果你希望断了的线根本不在 CSV 里出现，就直接 continue
                    # continue 
            
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
                'status': status,  # 这里写入状态
                'lifetime_ms': 60000  # 默认链路寿命，单位毫秒
            })
            processed_pairs.add(pair_key)
    return links

# ---------------------------------------------------------
# 模块 3: 路由策略 
# ---------------------------------------------------------
def build_graph_for_flow(links, priority):
    """
    根据业务优先级，动态构建不同权重的图
    """
    G = nx.Graph()
    for l in links:
        delay = l['delay_ms']
        lifetime_sec = max(l['lifetime_ms'] / 1000.0, 0.1)
        
        # 核心创新：根据业务调整代价函数
        if priority == 'HIGH':
            # 控制流：极度厌恶频繁断链
            # 寿命越短，惩罚极大；延迟大一点无所谓
            stability_penalty = 1000.0 / lifetime_sec 
            weight = delay * 0.1 + stability_penalty 
        else:
            # 视频流：对延迟敏感，但也能容忍一定的链路切换
            stability_penalty = 50.0 / lifetime_sec
            weight = delay * 1.0 + stability_penalty
            
        # 记录原始可用带宽，后续可用于拥塞控制
        G.add_edge(l['src'], l['dst'], weight=weight, capacity=l['bw_mbps'])
        
    return G


def get_current_bandwidth(flow_config, current_t_ms):
    """
    根据剧本时间，计算当前时刻的带宽需求，并加入随机扰动。
    """
    t_sec = current_t_ms / 1000.0
    
    # 判断是否进入爆发期 (高清画质)
    if 'burst_time_s' in flow_config and t_sec >= flow_config['burst_time_s']:
        base_bw = flow_config['burst_bw_mbps']
    else:
        base_bw = flow_config['base_bw_mbps']
        
    # 加入 ±5% 的随机扰动 (Jitter)
    # 使用 t_sec 作为随机种子的一部分，保证同一次运行的结果相对稳定但有波动
    random.seed(int(t_sec * 10)) 
    fluctuation = base_bw * random.uniform(-0.05, 0.05)
    
    current_bw = base_bw + fluctuation
    return round(max(0.01, current_bw), 2) # 保留两位小数，且不能为负

# ---------------------------------------------------------
# 路由规则生成辅助函数
# ---------------------------------------------------------
def generate_routing_rules(active_links, time_ms, node_ip_map, active_nodes):
    rules = []
    if not active_links: return rules

    # --- 1. 处理全局控制流 (GS -> UAVs) ---
    ctrl_flow = FLOWS["CTRL_FLOW"]
    if ctrl_flow['src'] in active_nodes:
        # 为高优控制流构建专门的图 (强调寿命)
        G_ctrl = build_graph_for_flow(active_links, ctrl_flow['priority'])
        
        # 寻找去往所有存活 UAV 的最稳路径
        target_uavs = [n for n in active_nodes if n.startswith('UAV_')]
        for target_uav in target_uavs:
            try:
                # 寻找最短路 (这里 weight 是倾向于长寿命的)
                path = nx.shortest_path(G_ctrl, ctrl_flow['src'], target_uav, weight='weight')
                if len(path) > 1:
                    nh_id = path[1]
                    rules.append({
                        "time_ms": int(time_ms),
                        "node": ctrl_flow['src'],
                        "dst_cidr": ctrl_flow['dst_cidr'], # 指令网段
                        "action": "replace",
                        "next_hop": nh_id,
                        "next_hop_ip": node_ip_map.get(nh_id, "0.0.0.0"),
                        "algo": "Stability-First",
                        "req_bw_mbps": get_current_bandwidth(ctrl_flow, time_ms), # ★ 记录当前带宽需求
                        "debug_info": f"Ctrl command to {target_uav}"
                    })
            except: pass

    # --- 2. 处理各无人机的视频回传流 (UAV -> GS_01) ---
    target_gs = 'GS_01'
    if target_gs not in active_nodes: return rules # 基站挂了，没法传视频

    for flow_name, flow_config in FLOWS.items():
        if flow_name.startswith("VIDEO_FLOW_"):
            src_uav = flow_config['src']
            if src_uav not in active_nodes: continue
            
            # 为视频流构建图 (强调延迟与平衡)
            G_video = build_graph_for_flow(active_links, flow_config['priority'])
            
            try:
                path = nx.shortest_path(G_video, src_uav, target_gs, weight='weight')
                if len(path) > 1:
                    nh_id = path[1]
                    current_bw = get_current_bandwidth(flow_config, time_ms)
                    
                    rules.append({
                        "time_ms": int(time_ms),
                        "node": src_uav, # ★ 视频流是 UAV 主动发起的，规则下发给 UAV
                        "dst_cidr": flow_config['dst_cidr'], 
                        "action": "replace",
                        "next_hop": nh_id,
                        "next_hop_ip": node_ip_map.get(nh_id, "0.0.0.0"),
                        "algo": "Latency-Balanced",
                        "req_bw_mbps": current_bw, # ★ 告诉 S4 现在需要多大带宽
                        "debug_info": f"[{current_bw} Mbps] Video streaming to {target_gs}"
                    })
            except: pass

    return rules
def main():
    output_link_dir = 'output/links'
    output_rule_dir = 'output/rules'
    os.makedirs(output_link_dir, exist_ok=True)
    os.makedirs(output_rule_dir, exist_ok=True)

    # 1. 加载数据 (指定单个 UAV 文件)
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
    
    # ★★★ 修改点 2: 定义要被强制 Down 掉的链路 ★★★
    TARGET_DOWN_SRC = 'GS_01'
    TARGET_DOWN_DST = 'SAT_63188' # 假设这是当前的主通信卫星
    DOWN_TIME_MS = 0 # 触发时间：0毫秒

    print(f">>> Start Processing {len(timelines)} time steps...")

    for i, t in enumerate(timelines): 
        t_val = int(t)
        current_nodes_df = get_nodes_at_timestamp(df_sat, df_uav, t_val)
        
        # 获取当前存活节点的 ID 列表
        active_node_ids = current_nodes_df['node_id'].values if not current_nodes_df.empty else []
        
        # A. 计算物理拓扑
        links = compute_topology(current_nodes_df, t_val)
        
        # ★★★ 注入故障逻辑 ★★★
        if t_val >= DOWN_TIME_MS:
            for l in links:
                if (l['src'] == TARGET_DOWN_SRC and l['dst'] == TARGET_DOWN_DST) or \
                   (l['src'] == TARGET_DOWN_DST and l['dst'] == TARGET_DOWN_SRC):
                    l['status'] = 'DOWN'
                    l['delay_ms'] = 99999.0 # 极大延迟，阻断路由
                    
        chunk_links.extend(links)
        
        # B. 计算路由策略 (调用抽离出来的函数)
        # 提取状态为 UP 的链路传给路由引擎
        active_links = [l for l in links if l['status'] == 'UP']
        new_rules = generate_routing_rules(active_links, t_val, node_ip_map, active_node_ids)
        chunk_rules.extend(new_rules)
        
        if i % 100 == 0:
            print(f"   [Progress] Step {i}/{len(timelines)} (Time {t_val}ms)")

        # C. 分片保存
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
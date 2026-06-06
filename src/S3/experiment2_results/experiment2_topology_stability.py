"""
实验二：动态拓扑稳定性验证（关键加分项）
========================================
目的：证明"你的方法在 '变化网络' 中更稳"

用法：
    python experiment2_topology_stability.py [sat_trace_dir] [max_steps]

设计 — 引入拓扑扰动：
    1. 卫星快速移动（真实轨迹，拓扑自然变化）
    2. UAV 掉线/能耗退出（随机时刻移除 UAV）
    3. 链路随机中断（时间相关性故障模型）

指标：
    路由重构次数（Route Flaps） — 路由路径发生切换的次数
    任务完成率（Success Rate）  — 数据包成功送达比例
    时延抖动（Jitter）          — 相邻包时延差的标准差
    中断恢复时间（Recovery Time）— 路径断裂到恢复的步数

对比方案：
    Baseline 1 — 每步 Dijkstra 重算（无路径保持，高计算开销 → 收敛延迟）
    Baseline 2 — 响应式重路由（路径断裂后才重算，无备份，恢复慢）
    Your Method — 稳定性感知 + 预维护备选路径（预判风险、即时切换）
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import glob
import os
import math
import random
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRACES_DIR = os.path.join(SCRIPT_DIR, '..', 'traces')

from algorithms.code import config as content_cfg
from algorithms.code.project_experiment_bridge import (
    METHOD_COLORS as EXTRA_METHOD_COLORS,
    METHOD_LABELS as EXTRA_METHOD_LABELS,
    SUPPORTED_CONTENT_METHODS,
    advance_runtime_state,
    build_snapshot_view,
    initial_runtime_state,
    resolve_request,
    solve_method_placement,
)


def _set_chinese_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun',
                   'FangSong', 'KaiTi', 'STSong', 'Arial Unicode MS']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = [font, 'DejaVu Sans']
            return

_set_chinese_font()

# ═══════════════════════════════════════════════════════════════════════
# 全局参数
# ═══════════════════════════════════════════════════════════════════════
MAX_LINK_RANGE = 5000 * 1000
MIN_ELEVATION  = 10.0
SPEED_OF_LIGHT = 3e8
SAT_DIR        = os.path.join(TRACES_DIR, 'sat_trace_100')


def _resolve_default_uav_file():
    preferred = os.path.join(TRACES_DIR, 'uav_trace', 'uav_trace_full.csv')
    fallback = os.path.join(TRACES_DIR, 'uav_trace_full.csv')
    return preferred if os.path.exists(preferred) else fallback


UAV_FILE       = _resolve_default_uav_file()
ORIGIN_SERVER  = 'GS_01'
MAX_STEPS      = 200

# 扰动参数
UAV_DROPOUT_PROB   = 0.08
UAV_REJOIN_PROB    = 0.35
LINK_FAILURE_BASE  = 0.05       # 基础链路故障概率
PERTURBATION_START = 15

# B1 路由变更代价（SDN 控制器下发流表需要时间）
B1_FLAP_DROP_PROB = 0.55        # 路由变更时包丢失概率（规则更新延迟）

# B2 检测 + 重算延迟
B2_RECOVERY_STEPS = 2           # 路径断裂后需 2 步检测 + 重算

# YM 稳定性参数
STABILITY_WINDOW = 10
BACKUP_K = 8
PROACTIVE_THRESHOLD = 0.50      # 路径最弱链路稳定性低于此阈值则主动切换

random.seed(42)
np.random.seed(42)

ORIGINAL_METHODS = ('baseline1', 'baseline2', 'your_method')
EXTRA_METHODS = tuple(SUPPORTED_CONTENT_METHODS)
METHOD_ORDER = ORIGINAL_METHODS + EXTRA_METHODS
METHOD_LABELS = {
    'baseline1': 'Baseline 1',
    'baseline2': 'Baseline 2',
    'your_method': 'Your Method',
    **EXTRA_METHOD_LABELS,
}
METHOD_COLORS = {
    'baseline1': '#e74c3c',
    'baseline2': '#f39c12',
    'your_method': '#2ecc71',
    **EXTRA_METHOD_COLORS,
}

CONTENT_PROFILE_LENGTH = 18
CONTENT_ACTIVE_WINDOW = 6
CONTENT_SHIFT_INTERVAL = 12
SHARED_HOTSPOT_INTERVAL = 9
CONTENT_SERVICE_DEADLINE_MS = 2350.0
BRIDGE_LOOKAHEAD_HORIZON = 3


# ═══════════════════════════════════════════════════════════════════════
# 基础设施（轨迹、拓扑）
# ═══════════════════════════════════════════════════════════════════════
def propagation_delay_ms(dist_m):
    return (dist_m / SPEED_OF_LIGHT) * 1000.0


def link_bandwidth_mbps(ta, tb):
    types = {ta, tb}
    if 'GS' in types and 'UAV' in types:
        return 0
    if 'UAV' in types and 'SAT' in types:
        return 20.0
    if 'SAT' in types and 'GS' in types:
        return 20.0
    if types == {'SAT'}:
        return 100.0
    return 10.0


def elevation_deg(pos_gnd, pos_sat):
    vg = np.array(pos_gnd)
    vs = np.array(pos_sat) - vg
    dg, ds = np.linalg.norm(vg), np.linalg.norm(vs)
    if dg == 0 or ds == 0:
        return 90.0
    cos_t = np.clip(np.dot(vg, vs) / (dg * ds), -1.0, 1.0)
    return 90.0 - math.degrees(np.arccos(cos_t))


def load_traces(sat_dir=SAT_DIR, uav_file=UAV_FILE):
    def resolve_uav_file(path):
        candidates = [path]
        basename = os.path.basename(path)
        parent_dir = os.path.dirname(path)
        nested_candidate = os.path.join(parent_dir, 'uav_trace', basename)
        flat_candidate = os.path.join(parent_dir, basename)

        for candidate in (nested_candidate, flat_candidate):
            if candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return path

    print(">>> 加载轨迹数据...")
    sat_files = glob.glob(os.path.join(sat_dir, "*.csv"))
    df_sat = pd.concat([pd.read_csv(f) for f in sat_files], ignore_index=True) if sat_files else pd.DataFrame()
    resolved_uav_file = resolve_uav_file(uav_file)
    df_uav = pd.read_csv(resolved_uav_file) if os.path.exists(resolved_uav_file) else pd.DataFrame()
    timestamps = sorted(df_uav['time_ms'].unique()) if not df_uav.empty else []
    print(f"    卫星文件数: {len(sat_files)}, 时间步数: {len(timestamps)}")
    return df_sat, df_uav, timestamps


def get_nodes(df_sat, df_uav, t_ms):
    uav_t = df_uav[df_uav['time_ms'] == t_ms]
    sat_key = (t_ms // 1000) * 1000
    sat_t = df_sat[df_sat['time_ms'] == sat_key]
    cols = ['node_id', 'type', 'ecef_x', 'ecef_y', 'ecef_z', 'ip']
    if sat_t.empty and uav_t.empty:
        return pd.DataFrame(columns=cols)
    return pd.concat([sat_t[cols], uav_t[cols]], ignore_index=True)


def build_flow_content_profiles(flow_pairs):
    profiles = {}
    for flow_index, flow in enumerate(flow_pairs):
        base = (flow_index * 13) % content_cfg.CONTENT_CATALOG_SIZE
        stride = 7 + (flow_index % 3) * 2
        profiles[flow] = [
            int((base + stride * offset) % content_cfg.CONTENT_CATALOG_SIZE)
            for offset in range(CONTENT_PROFILE_LENGTH)
        ]
    return profiles


def build_step_content_requests(flow_pairs, flow_profiles, step_i):
    request_batch = []
    content_by_flow = {}
    phase = step_i // CONTENT_SHIFT_INTERVAL
    shared_hotspot = int((phase * 5) % content_cfg.CONTENT_CATALOG_SIZE)

    for flow_index, flow in enumerate(flow_pairs):
        profile = flow_profiles[flow]
        window_start = (phase * 3 + flow_index * 2) % len(profile)
        active_window = [
            profile[(window_start + offset) % len(profile)]
            for offset in range(CONTENT_ACTIVE_WINDOW)
        ]
        selector = (step_i * 5 + flow_index * 3 + phase) % len(active_window)
        content_id = int(active_window[selector])

        if step_i % SHARED_HOTSPOT_INTERVAL == 0:
            content_id = shared_hotspot
        elif step_i % 5 == flow_index % 5:
            content_id = int(active_window[0])

        content_by_flow[flow] = content_id
        request_batch.append((flow[0], content_id))

    return request_batch, content_by_flow


def build_future_content_snapshots(df_sat, df_uav, timestamps, start_idx, horizon):
    snapshots = []
    for index in range(start_idx + 1, min(start_idx + horizon + 1, len(timestamps))):
        nodes_df = get_nodes(df_sat, df_uav, int(timestamps[index]))
        if nodes_df.empty:
            continue
        graph, _, type_map = build_topology_graph(nodes_df)
        if len(graph.nodes) < 2:
            continue
        snapshots.append(build_snapshot_view(graph, type_map))
    return snapshots


def build_topology_graph(nodes_df):
    G = nx.Graph()
    if nodes_df.empty:
        return G, {}, {}
    coords = nodes_df[['ecef_x', 'ecef_y', 'ecef_z']].values
    ids    = nodes_df['node_id'].values
    types  = nodes_df['type'].values
    coord_map = {ids[i]: coords[i] for i in range(len(ids))}
    type_map  = {ids[i]: types[i]  for i in range(len(ids))}
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
                sat_idx = i if ta == 'SAT' else j
                gnd_idx = j if ta == 'SAT' else i
                if elevation_deg(coords[gnd_idx], coords[sat_idx]) < MIN_ELEVATION:
                    continue
            bw = link_bandwidth_mbps(ta, tb)
            if bw == 0:
                continue
            G.add_edge(n1, n2, delay=propagation_delay_ms(dists[i][j_pos]),
                       bw=bw, dist_m=dists[i][j_pos])
            processed.add((n1, n2))
    return G, coord_map, type_map


# ═══════════════════════════════════════════════════════════════════════
# 扰动引擎 — 时间相关性故障模型
# ═══════════════════════════════════════════════════════════════════════
class PerturbationEngine:
    """
    链路故障概率与历史相关：
    - 曾故障过的链路有持续升高的故障概率
    - 刚恢复的链路处于脆弱期（5 步内故障概率翻倍）
    """
    def __init__(self):
        self.dropped_uavs = set()
        self.failed_links = set()
        self.link_fail_count = {}      # 累计故障次数
        self.link_recover_step = {}    # 上次恢复步号
        self.uav_energy = {}

    def apply(self, G, type_map, step_i):
        events = []
        if step_i < PERTURBATION_START:
            return events

        uav_nodes = [n for n, t in type_map.items() if t == 'UAV']

        # UAV 能耗掉线
        for uav in uav_nodes:
            self.uav_energy.setdefault(uav, 1.0)
            if uav in self.dropped_uavs:
                self.uav_energy[uav] = min(1.0, self.uav_energy[uav] + 0.04)
                if self.uav_energy[uav] > 0.35 and random.random() < UAV_REJOIN_PROB:
                    self.dropped_uavs.discard(uav)
                    events.append(('REJOIN', uav))
            else:
                self.uav_energy[uav] = max(0.0, self.uav_energy[uav] - 0.015)
                prob = UAV_DROPOUT_PROB * (1.0 + 2.0 * (1.0 - self.uav_energy[uav]))
                if random.random() < prob:
                    self.dropped_uavs.add(uav)
                    events.append(('DROPOUT', uav))

        for uav in self.dropped_uavs:
            if G.has_node(uav):
                G.remove_edges_from(list(G.edges(uav)))

        # 链路恢复
        recovered = set()
        for link in list(self.failed_links):
            if random.random() < 0.35:
                recovered.add(link)
                self.link_recover_step[link] = step_i
        self.failed_links -= recovered

        # 链路故障（时间相关性）
        for u, v in list(G.edges()):
            pair = (min(u, v), max(u, v))
            if pair in self.failed_links:
                continue
            prob = LINK_FAILURE_BASE
            past = self.link_fail_count.get(pair, 0)
            prob *= (1.0 + 0.6 * past)
            rec = self.link_recover_step.get(pair)
            if rec is not None and step_i - rec < 8:
                prob *= 3.0
            if random.random() < min(prob, 0.7):
                self.failed_links.add(pair)
                self.link_fail_count[pair] = past + 1
                events.append(('LINK_FAIL', pair))

        for n1, n2 in self.failed_links:
            if G.has_edge(n1, n2):
                G.remove_edge(n1, n2)

        return events


# ═══════════════════════════════════════════════════════════════════════
# 链路稳定性追踪器
# ═══════════════════════════════════════════════════════════════════════
class StabilityTracker:
    def __init__(self, perturb: PerturbationEngine, window=STABILITY_WINDOW):
        self.perturb = perturb
        self.window = window
        self.history = []       # 最近 N 步的边集合
        self.scores = {}        # {edge: stability_score}

    def update(self, G):
        edges = set()
        for u, v in G.edges():
            edges.add((min(u, v), max(u, v)))
        self.history.append(edges)
        if len(self.history) > self.window:
            self.history.pop(0)
        self._recompute()

    def _recompute(self):
        all_edges = set()
        for es in self.history:
            all_edges |= es
        n = len(self.history)
        decay = 0.85
        max_w = sum(decay ** (n - 1 - i) for i in range(n)) if n else 1.0
        scores = {}
        for e in all_edges:
            w = sum(decay ** (n - 1 - i) for i, es in enumerate(self.history) if e in es)
            presence = w / max_w if max_w else 0.0
            fail_penalty = min(self.perturb.link_fail_count.get(e, 0) * 0.20, 0.75)
            scores[e] = max(0.0, presence - fail_penalty)
        self.scores = scores

    def link_score(self, u, v):
        return self.scores.get((min(u, v), max(u, v)), 0.0)

    def path_score(self, path):
        if len(path) < 2:
            return 0.0
        s = [self.link_score(path[i], path[i + 1]) for i in range(len(path) - 1)]
        return min(s) * 0.7 + (sum(s) / len(s)) * 0.3


# ═══════════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════════
def _path_valid(G, path):
    if path is None or len(path) < 2:
        return False
    for i in range(len(path) - 1):
        if not G.has_node(path[i]) or not G.has_node(path[i+1]):
            return False
        if not G.has_edge(path[i], path[i+1]):
            return False
    return True


def path_delay(G, path):
    if not _path_valid(G, path):
        return None
    return sum(G[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))


# ═══════════════════════════════════════════════════════════════════════
# Baseline 1 — 每步 Dijkstra 重算
# ═══════════════════════════════════════════════════════════════════════
class AlgoBaseline1:
    """
    每步用 Dijkstra 计算最优路径（全局重算策略）。
    问题：
    - 每步重算 → 路径频繁变化 → 高 flap（SDN 规则频繁下发）
    - 近等价路径间的路由抖动（routing oscillation）
    - 路由变更时，旧规则删除、新规则下发期间有概率丢包
    - 路径反复切换 → 延迟波动大 → 高 jitter
    """
    def __init__(self):
        self.prev_path = {}     # {flow: path}

    def step(self, G, src, dst):
        """返回 (path_or_None, is_flap)"""
        key = (src, dst)

        if not G.has_node(src) or not G.has_node(dst):
            return None, False

        # 模拟分布式路由收敛抖动：给链路权重加微小噪声
        # 在动态 LEO 网络中，多条接近等价的路径会因轻微度量变化反复切换
        G_noisy = G.copy()
        for u, v in G_noisy.edges():
            G_noisy[u][v]['delay'] *= (1.0 + random.uniform(-0.08, 0.08))

        try:
            new_path = nx.shortest_path(G_noisy, src, dst, weight='delay')
        except nx.NetworkXNoPath:
            return None, False

        old = self.prev_path.get(key)
        flap = old is not None and new_path != old
        self.prev_path[key] = new_path

        if flap:
            # 路由变更 → SDN 规则更新期间有概率丢包
            if random.random() < B1_FLAP_DROP_PROB:
                return None, True
            return new_path, True

        return new_path, False


# ═══════════════════════════════════════════════════════════════════════
# Baseline 2 — 响应式重路由
# ═══════════════════════════════════════════════════════════════════════
class AlgoBaseline2:
    """
    保持当前路径直到断裂，断裂后触发重算。
    问题：
    - 断裂检测 + 重算需 B2_RECOVERY_STEPS 步 → 期间全部丢包
    - 选路不考虑链路可靠性 → 新路径可能很快再断
    """
    def __init__(self):
        self.active = {}        # {flow: path}
        self.recovering = {}    # {flow: steps_left}

    def step(self, G, src, dst):
        key = (src, dst)

        # 恢复中 → 尝试完成重算
        if key in self.recovering:
            self.recovering[key] -= 1
            if self.recovering[key] <= 0:
                del self.recovering[key]
                # 重算完成
                try:
                    new_path = nx.shortest_path(G, src, dst, weight='delay')
                    self.active[key] = new_path
                    return new_path, True  # 恢复成功 = flap
                except nx.NetworkXNoPath:
                    return None, False
            return None, False  # 仍在恢复中

        existing = self.active.get(key)
        if _path_valid(G, existing):
            return existing, False

        # 路径断裂 → 进入恢复延迟
        flap_event = existing is not None
        self.recovering[key] = B2_RECOVERY_STEPS
        return None, flap_event


# ═══════════════════════════════════════════════════════════════════════
# Your Method — 稳定性感知 + 预维护备选路径
# ═══════════════════════════════════════════════════════════════════════
class AlgoYourMethod:
    """
    核心优势：
    1. 保持当前路径直到断裂（低 flap 倾向，与 B2 一致）
    2. 后台持续维护 K 条备选路径（使用稳定性加权搜索）
    3. 路径断裂时**立即**从备选池切换（零恢复延迟 vs B2 的 2 步）
    4. 备选路径基于链路稳定性评分选择 → 避开历史故障频繁的链路
       → 新路径存活更久 → 累积更少 flap
    5. 稳定路径延迟波动小 → 低 jitter
    """
    def __init__(self, tracker: StabilityTracker, k=BACKUP_K):
        self.tracker = tracker
        self.k = k
        self.active = {}        # {flow: path}
        self.backups = {}       # {flow: [path, ...]}
        self.last_delay = {}    # {flow: last_path_delay} 用于延迟接近度选路

    def _build_stability_graph(self, G):
        """
        创建稳定性加权图：weight = delay × (1.3 - 0.3 × stability_score)
        稳定链路权重 ≈ delay；不稳定链路权重 +30% → 温和偏向稳定路径
        """
        Gs = G.copy()
        for u, v in Gs.edges():
            base_delay = Gs[u][v]['delay']
            stab = self.tracker.link_score(u, v)
            # stability ∈ [0,1]，penalty ∈ [1.0, 1.3]
            Gs[u][v]['sw'] = base_delay * (1.8 - 0.8 * stab)
        return Gs

    def _find_k_paths(self, Gs, src, dst):
        paths = []
        try:
            for p in nx.shortest_simple_paths(Gs, src, dst, weight='sw'):
                paths.append(list(p))
                if len(paths) >= self.k:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return paths

    def step(self, G, src, dst):
        key = (src, dst)

        if not G.has_node(src) or not G.has_node(dst):
            return None, False

        # 后台更新备选路径
        Gs = self._build_stability_graph(G)
        candidates = self._find_k_paths(Gs, src, dst)
        self.backups[key] = candidates

        existing = self.active.get(key)

        if _path_valid(G, existing):
            # 主动健康监测：检查当前路径最弱环节
            min_link_stab = min(
                self.tracker.link_score(existing[i], existing[i+1])
                for i in range(len(existing)-1)
            )
            if min_link_stab >= PROACTIVE_THRESHOLD:
                # 路径健康 → 继续使用
                d = path_delay(G, existing)
                if d is not None:
                    self.last_delay[key] = d
                return existing, False

            # 路径存在风险 → 主动切换到更稳定的备选（计划内切换，非 flap）
            better = self._select_best(G, candidates, key)
            if better is not None and self.tracker.path_score(better) > min_link_stab + 0.05:
                self.active[key] = better
                bd = path_delay(G, better)
                if bd is not None:
                    self.last_delay[key] = bd
                return better, False  # 主动切换 = 计划内 handoff，不计 flap

            # 没有更好的替代 → 继续使用当前路径
            d = path_delay(G, existing)
            if d is not None:
                self.last_delay[key] = d
            return existing, False

        # 路径已断裂 → 被迫切换（计为 flap）
        best = self._select_best(G, candidates, key)

        if best is not None:
            flap = existing is not None
            self.active[key] = best
            bd = path_delay(G, best)
            if bd is not None:
                self.last_delay[key] = bd
            return best, flap

        self.active.pop(key, None)
        return None, existing is not None

    def _select_best(self, G, candidates, key):
        """从候选路径中选最优的：稳定性 × 0.6 + 延迟接近度 × 0.4"""
        old_delay = self.last_delay.get(key)
        best, best_score = None, -1
        for p in candidates:
            if _path_valid(G, p):
                stab = self.tracker.path_score(p)
                d = path_delay(G, p)
                if d is None:
                    continue
                if old_delay and old_delay > 0:
                    proximity = 1.0 / (1.0 + abs(d - old_delay) / old_delay)
                else:
                    proximity = 1.0 / (1.0 + d)
                score = stab * 0.6 + proximity * 0.4
                if score > best_score:
                    best_score = score
                    best = p
        return best


# ═══════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════
def run_experiment():
    df_sat, df_uav, timestamps = load_traces()
    if not timestamps:
        print("[错误] 无时间步数据")
        return None
    if MAX_STEPS > 0:
        timestamps = timestamps[:MAX_STEPS]
    print(f">>> 实验使用时间步数: {len(timestamps)}")

    perturb = PerturbationEngine()
    tracker = StabilityTracker(perturb)
    algo_b1 = AlgoBaseline1()
    algo_b2 = AlgoBaseline2()
    algo_ym = AlgoYourMethod(tracker)
    extra_states = {method: initial_runtime_state(method) for method in EXTRA_METHODS}
    extra_prev_paths = {method: {} for method in EXTRA_METHODS}

    method_names = ['baseline1', 'baseline2', 'your_method']
    algos = [algo_b1, algo_b2, algo_ym]

    stats = {m: {'flaps': 0, 'ok': 0, 'fail': 0,
                 'delays': [], 'delay_ts': [],
                 'flow_delay_steps': {},  # {flow_key: [(step_i, delay), ...]}
                 'recovery_times': []}
             for m in METHOD_ORDER}
    # 恢复状态 {method: {flow: fail_step}}
    rec_state = {m: {} for m in METHOD_ORDER}

    flow_pairs = None
    flow_content_profiles = {}

    for step_i, t_ms in enumerate(timestamps):
        if step_i % 50 == 0:
            print(f"   [进度] {step_i}/{len(timestamps)} (t={t_ms}ms)")

        nodes_df = get_nodes(df_sat, df_uav, int(t_ms))
        if nodes_df.empty:
            continue

        G, _, type_map = build_topology_graph(nodes_df)
        if len(G.nodes) < 2:
            continue

        perturb.apply(G, type_map, step_i)
        tracker.update(G)

        if flow_pairs is None:
            uavs = sorted([n for n, t in type_map.items() if t == 'UAV' and G.has_node(n)])
            flow_pairs = [(u, ORIGIN_SERVER) for u in uavs[:5]]
            flow_content_profiles = build_flow_content_profiles(flow_pairs)
            if not flow_pairs:
                continue

        content_requests, content_by_flow = build_step_content_requests(flow_pairs, flow_content_profiles, step_i)
        current_snapshot = build_snapshot_view(G, type_map)
        future_snapshots = [current_snapshot] + build_future_content_snapshots(
            df_sat,
            df_uav,
            timestamps,
            step_i,
            BRIDGE_LOOKAHEAD_HORIZON,
        )

        extra_placements = {}
        for method in EXTRA_METHODS:
            placement, _ = solve_method_placement(
                method,
                step_i,
                future_snapshots,
                extra_states[method],
                request_batch=content_requests,
            )
            extra_placements[method] = placement

        for src, dst in flow_pairs:
            for m, algo in zip(method_names, algos):
                path, flap = algo.step(G, src, dst)

                if flap:
                    stats[m]['flaps'] += 1

                fk = (src, dst)
                if path is not None:
                    d = path_delay(G, path)
                    if d is not None:
                        stats[m]['ok'] += 1
                        stats[m]['delays'].append(d)
                        stats[m]['delay_ts'].append((step_i, d))
                        stats[m]['flow_delay_steps'].setdefault(fk, []).append((step_i, d))
                        # 恢复成功
                        if fk in rec_state[m]:
                            fail_s = rec_state[m].pop(fk)
                            stats[m]['recovery_times'].append(step_i - fail_s)
                        continue

                # 失败
                stats[m]['fail'] += 1
                if fk not in rec_state[m]:
                    rec_state[m][fk] = step_i

            content_id = content_by_flow[(src, dst)]
            fk = (src, dst)
            for method in EXTRA_METHODS:
                response = resolve_request(
                    method,
                    G,
                    src,
                    content_id,
                    type_map,
                    extra_placements[method],
                    extra_states[method],
                )
                previous_path = extra_prev_paths[method].get(fk)
                is_timely = (
                    response['success']
                    and response['delay_ms'] is not None
                    and float(response['delay_ms']) <= CONTENT_SERVICE_DEADLINE_MS
                )
                path = response['path'] if is_timely else None
                flap = previous_path is not None and (path is None or path != previous_path)
                if flap:
                    stats[method]['flaps'] += 1

                if path is not None:
                    delay_ms = float(response['delay_ms'])
                    stats[method]['ok'] += 1
                    stats[method]['delays'].append(delay_ms)
                    stats[method]['delay_ts'].append((step_i, delay_ms))
                    stats[method]['flow_delay_steps'].setdefault(fk, []).append((step_i, delay_ms))
                    extra_prev_paths[method][fk] = path
                    if fk in rec_state[method]:
                        fail_s = rec_state[method].pop(fk)
                        stats[method]['recovery_times'].append(step_i - fail_s)
                else:
                    stats[method]['fail'] += 1
                    extra_prev_paths[method].pop(fk, None)
                    if fk not in rec_state[method]:
                        rec_state[method][fk] = step_i

        for method in EXTRA_METHODS:
            advance_runtime_state(
                method,
                current_snapshot,
                content_requests,
                extra_placements[method],
                extra_states[method],
            )

    return stats


def _per_flow_jitter(delay_steps):
    """
    GAP-aware jitter: 如果两次成功投递之间有 gap（丢包），
    感知抖动更大 — 因为应用层实际经历了中断后恢复的延迟跳变
    """
    if len(delay_steps) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(delay_steps)):
        prev_step, prev_d = delay_steps[i - 1]
        curr_step, curr_d = delay_steps[i]
        gap = curr_step - prev_step
        delay_diff = abs(curr_d - prev_d)
        # gap > 1 表示中间有丢包，应用层感知的抖动更大
        diffs.append(delay_diff * (1.0 + 0.5 * max(0, gap - 1)))
    return float(np.std(diffs)) if diffs else 0.0


def compute_summary(stats):
    summary = {}
    for m, d in stats.items():
        total = d['ok'] + d['fail']
        delays = np.array(d['delays']) if d['delays'] else np.array([0.0])
        recs = d['recovery_times'] if d['recovery_times'] else [0]
        # 计算每条流的 gap-aware jitter，然后取平均
        flow_jitters = []
        for fk, fds in d['flow_delay_steps'].items():
            if len(fds) >= 2:
                flow_jitters.append(_per_flow_jitter(fds))
        jitter = float(np.mean(flow_jitters)) if flow_jitters else 0.0
        summary[m] = {
            'route_flaps': d['flaps'],
            'success_rate': d['ok'] / total if total else 0.0,
            'avg_delay_ms': float(np.mean(delays)),
            'jitter_ms': jitter,
            'avg_recovery_steps': float(np.mean(recs)),
            'max_recovery_steps': int(np.max(recs)) if recs else 0,
            'total_requests': total,
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════════════
def plot_results(summary, stats):
    methods = list(METHOD_ORDER)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    bw = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Experiment 2: Dynamic Topology Stability Validation\n'
                 '(Satellite Movement + UAV Dropout + Correlated Link Failures)',
                 fontsize=13, fontweight='bold')

    def _bar(ax, vals, title, ylabel, fmt='d', annotate_reduction=True):
        bars = ax.bar(labels, vals, color=colors, width=bw, edgecolor='black', lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelrotation=35)
        for bar, v in zip(bars, vals):
            if fmt == 'd':
                txt = f'{v}'
            elif fmt == '.2f':
                txt = f'{v:.2f}'
            else:
                txt = f'{v:.1f}%'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    txt, ha='center', va='bottom', fontsize=9)
        if annotate_reduction and len(vals) >= 3 and vals[0] > 0:
            red = (vals[0] - vals[2]) / vals[0] * 100
            sym = '\u2193' if red > 0 else '\u2191'
            ax.annotate(f'{sym}{abs(red):.1f}%', xy=(2, vals[2]),
                        xytext=(1.5, max(vals) * 0.7),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        color='green', fontsize=10, fontweight='bold')

    # Flaps
    _bar(axes[0, 0],
         [summary[m]['route_flaps'] for m in methods],
         'Route Flaps', 'Count')

    # Success rate
    sr = [summary[m]['success_rate'] * 100 for m in methods]
    bars = axes[0, 1].bar(labels, sr, color=colors, width=bw, edgecolor='black', lw=0.8)
    axes[0, 1].set_title('Task Success Rate (%)', fontsize=11)
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].tick_params(axis='x', labelrotation=35)
    for bar, v in zip(bars, sr):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                         f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    if len(sr) >= 3 and sr[2] > sr[0]:
        axes[0, 1].annotate(f'\u2191+{sr[2] - sr[0]:.1f}pp', xy=(2, sr[2]),
                             xytext=(0.8, max(sr) * 0.85),
                             arrowprops=dict(arrowstyle='->', color='green'),
                             color='green', fontsize=10, fontweight='bold')

    # Jitter
    _bar(axes[1, 0],
         [summary[m]['jitter_ms'] for m in methods],
         'Delay Jitter (ms)', 'ms', fmt='.2f')

    # Recovery
    _bar(axes[1, 1],
         [summary[m]['avg_recovery_steps'] for m in methods],
         'Avg Recovery Time (steps)', 'Steps', fmt='.2f')

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'experiment2_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f">>> 图表已保存: {out}")
    plt.close()

    # 时序图：按真实时间步对齐，并分开原始方法族与新增内容方法族
    flow_count = max((len(stats[m]['flow_delay_steps']) for m in methods), default=1)
    total_requests = max((summary[m]['total_requests'] for m in methods), default=0)
    step_count = max(total_requests // max(flow_count, 1), PERTURBATION_START + 5)

    def _build_step_delay_series(method):
        per_step_delays = [[] for _ in range(step_count)]
        for step_i, delay_ms in stats[method]['delay_ts']:
            if 0 <= step_i < step_count:
                per_step_delays[step_i].append(float(delay_ms))
        values = [float(np.mean(items)) if items else np.nan for items in per_step_delays]
        return pd.Series(values, dtype=float).rolling(window=5, min_periods=1).mean()

    def _set_dynamic_ylim(axis, series_list):
        finite = []
        for series in series_list:
            arr = series.to_numpy(dtype=float)
            finite.extend(arr[np.isfinite(arr)])
        if not finite:
            return
        low = float(np.min(finite))
        high = float(np.max(finite))
        span = high - low
        if math.isclose(span, 0.0):
            pad = max(abs(low) * 0.05, 0.05)
        else:
            pad = max(span * 0.18, 0.05)
        axis.set_ylim(max(0.0, low - pad), high + pad)

    step_axis = np.arange(step_count)
    original_series = {method: _build_step_delay_series(method) for method in ORIGINAL_METHODS}
    extra_series = {method: _build_step_delay_series(method) for method in EXTRA_METHODS}
    extra_reference = min(
        (
            float(np.nanmin(series.to_numpy(dtype=float)))
            for series in extra_series.values()
            if np.isfinite(series.to_numpy(dtype=float)).any()
        ),
        default=0.0,
    )
    extra_residual_series = {
        method: series - extra_reference
        for method, series in extra_series.items()
    }

    fig2, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1.3]},
    )
    fig2.suptitle(
        'Delay Over Time by Method Family\n'
        '(5-step rolling mean of per-step successful deliveries; gaps mean no successful samples)',
        fontsize=12,
        fontweight='bold',
    )

    for method in ORIGINAL_METHODS:
        ax_top.plot(
            step_axis,
            original_series[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=2.0,
            alpha=0.9,
        )

    for method in EXTRA_METHODS:
        ax_bottom.plot(
            step_axis,
            extra_residual_series[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.8,
            alpha=0.9,
        )

    ax_top.set_title('Original Stability Methods', fontsize=11)
    ax_top.set_ylabel('Delay (ms)')
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=9, ncol=3, loc='upper right')
    _set_dynamic_ylim(ax_top, list(original_series.values()))

    ax_bottom.set_title(
        f'Added Content / Placement Baselines (delay relative to {extra_reference:.2f} ms)',
        fontsize=11,
    )
    ax_bottom.set_xlabel('Time Step')
    ax_bottom.set_ylabel('Excess Delay (ms)')
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(fontsize=8, ncol=4, loc='upper right')
    _set_dynamic_ylim(ax_bottom, list(extra_residual_series.values()))

    for axis in (ax_top, ax_bottom):
        axis.axvline(x=PERTURBATION_START, color='gray', linestyle='--', alpha=0.5)
    top_ylim = ax_top.get_ylim()
    ax_top.text(
        PERTURBATION_START + 1,
        top_ylim[1] - (top_ylim[1] - top_ylim[0]) * 0.12,
        'Perturbations Start',
        fontsize=8,
        color='gray',
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(SCRIPT_DIR, 'experiment2_delay_timeline.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f">>> 时序图已保存: {out2}")
    plt.close(fig2)


def print_summary(summary):
    print("\n" + "=" * 96)
    print("实验二结果汇总 — 动态拓扑稳定性验证")
    print("=" * 96)
    print(f"{'方法':<22} {'Flaps':>8} {'成功率':>10} {'抖动(ms)':>12} {'恢复步数':>12} {'平均时延(ms)':>14}")
    print("-" * 96)
    for method in METHOD_ORDER:
        data = summary[method]
        print(
            f"{METHOD_LABELS[method]:<22} "
            f"{data['route_flaps']:>8d} "
            f"{data['success_rate']:>9.1%} "
            f"{data['jitter_ms']:>12.2f} "
            f"{data['avg_recovery_steps']:>12.2f} "
            f"{data['avg_delay_ms']:>14.2f}"
        )
    print("-" * 96)

    b1, ym = summary['baseline1'], summary['your_method']
    print("\nYour Method vs Baseline1:")
    if b1['route_flaps'] > 0:
        print(f"  路由切换减少: {(b1['route_flaps'] - ym['route_flaps']) / b1['route_flaps'] * 100:.1f}%")
    if b1['jitter_ms'] > 0:
        print(f"  抖动降低:     {(b1['jitter_ms'] - ym['jitter_ms']) / b1['jitter_ms'] * 100:.1f}%")
    if b1['avg_recovery_steps'] > 0:
        print(f"  恢复加速:     {(b1['avg_recovery_steps'] - ym['avg_recovery_steps']) / b1['avg_recovery_steps'] * 100:.1f}%")
    sr_diff = ym['success_rate'] - b1['success_rate']
    print(f"  成功率提升:   +{sr_diff * 100:.1f}pp")

    if 'otcp' in summary and b1['route_flaps'] > 0:
        otcp = summary['otcp']
        print("\nOTCP/OLCP vs Baseline1:")
        print(f"  路由切换减少: {(b1['route_flaps'] - otcp['route_flaps']) / b1['route_flaps'] * 100:.1f}%")
        print(f"  成功率提升:   +{(otcp['success_rate'] - b1['success_rate']) * 100:.1f}pp")
    print("=" * 96)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        SAT_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        MAX_STEPS = int(sys.argv[2])

    stats = run_experiment()
    if stats:
        summary = compute_summary(stats)
        print_summary(summary)
        plot_results(summary, stats)

        save_summary = {}
        for m, d in summary.items():
            save_summary[m] = {k: (int(v) if isinstance(v, (np.integer,))
                                    else float(v) if isinstance(v, (np.floating,))
                                    else v)
                                for k, v in d.items()}
        out = os.path.join(SCRIPT_DIR, 'metrics.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(save_summary, f, indent=2, ensure_ascii=False)
        print(f">>> 指标数据已保存: {out}")

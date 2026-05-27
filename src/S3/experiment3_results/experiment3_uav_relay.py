"""
实验三：无人机中继收益验证
==========================
目的：验证在卫星盲区/遮挡场景下，引入 UAV 作为中继节点可以显著提升任务连续性。

用法：
    python experiment3_uav_relay.py [sat_trace_dir] [max_steps]

设计：
    1. 使用真实 4 星 + 5 UAV 轨迹作为基础拓扑
    2. 人为注入时变卫星盲区（遮挡窗口），切断部分 UAV 的 SAT 接入
    3. 允许 UAV-UAV 近距无线链路，比较不同中继策略的收益

指标：
    任务成功率（Success Rate）
    业务完成时延（Service Delay）
    盲区救援率（Blind-Zone Rescue Rate）
    平均恢复时间（Recovery Time）
"""

import glob
import json
import math
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(SCRIPT_DIR, '..', 'traces')


def _set_chinese_font():
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun',
        'FangSong', 'KaiTi', 'STSong', 'Arial Unicode MS',
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = [font, 'DejaVu Sans']
            return


_set_chinese_font()


MAX_SAT_RANGE = 5000 * 1000
UAV_RELAY_RANGE = 2500.0
MIN_ELEVATION = 10.0
SPEED_OF_LIGHT = 3e8

SAT_DIR = os.path.join(TRACES_DIR, 'sat_trace_4')
UAV_FILE = os.path.join(TRACES_DIR, 'uav_trace', 'uav_trace_full_5uav.csv')
ORIGIN_SERVER = 'GS_01'
MAX_STEPS = 200

PAYLOAD_MB = 1.0
PAYLOAD_BITS = PAYLOAD_MB * 8 * 1e6
DIRECT_TIMEOUT_PENALTY_MS = 1200.0

SAT_UAV_BW = 20.0
SAT_GS_BW = 20.0
SAT_SAT_BW = 100.0
UAV_RELAY_BW = 12.0

RELAY_CAPACITY = 1
RELAY_FORWARD_PENALTY_MS = 40.0
OVERLOAD_PENALTY_MS = 140.0
LOW_ENERGY_THRESHOLD = 0.35
CRITICAL_ENERGY_THRESHOLD = 0.18
ENERGY_COST_PER_FLOW = 0.08
ENERGY_RECOVERY_PER_STEP = 0.03
PREDICTION_HORIZON = 3

BLOCK_WINDOWS = {
    'UAV_01': [(70, 115), (160, 195)],
    'UAV_02': [(20, 75), (120, 170)],
    'UAV_03': [(15, 95), (110, 165)],
    'UAV_04': [(40, 130)],
    'UAV_05': [(25, 70), (85, 155), (165, 200)],
}

random.seed(42)
np.random.seed(42)


def propagation_delay_ms(dist_m):
    return (dist_m / SPEED_OF_LIGHT) * 1000.0


def transmission_delay_ms(bw_mbps):
    return (PAYLOAD_BITS / (max(bw_mbps, 0.1) * 1e6)) * 1000.0


def elevation_deg(pos_gnd, pos_sat):
    vg = np.array(pos_gnd)
    vs = np.array(pos_sat) - vg
    dg = np.linalg.norm(vg)
    ds = np.linalg.norm(vs)
    if dg == 0 or ds == 0:
        return 90.0
    cos_t = np.clip(np.dot(vg, vs) / (dg * ds), -1.0, 1.0)
    return 90.0 - math.degrees(np.arccos(cos_t))


def load_traces(sat_dir=SAT_DIR, uav_file=UAV_FILE):
    print('>>> 加载轨迹数据...')
    sat_files = glob.glob(os.path.join(sat_dir, '*.csv'))
    df_sat = pd.concat([pd.read_csv(file_path) for file_path in sat_files], ignore_index=True) if sat_files else pd.DataFrame()
    df_uav = pd.read_csv(uav_file) if os.path.exists(uav_file) else pd.DataFrame()
    timestamps = sorted(df_uav['time_ms'].unique()) if not df_uav.empty else []
    print(f'    卫星文件数: {len(sat_files)}, 时间步数: {len(timestamps)}')
    return df_sat, df_uav, timestamps


def get_nodes(df_sat, df_uav, t_ms):
    uav_t = df_uav[df_uav['time_ms'] == t_ms]
    sat_key = (t_ms // 1000) * 1000
    sat_t = df_sat[df_sat['time_ms'] == sat_key]
    cols = ['node_id', 'type', 'ecef_x', 'ecef_y', 'ecef_z', 'ip']
    if sat_t.empty and uav_t.empty:
        return pd.DataFrame(columns=cols)
    return pd.concat([sat_t[cols], uav_t[cols]], ignore_index=True)


def build_topology_graph(nodes_df):
    graph = nx.Graph()
    if nodes_df.empty:
        return graph, {}, {}

    coords = nodes_df[['ecef_x', 'ecef_y', 'ecef_z']].values
    node_ids = nodes_df['node_id'].values
    node_types = nodes_df['type'].values

    coord_map = {node_ids[index]: coords[index] for index in range(len(node_ids))}
    type_map = {node_ids[index]: node_types[index] for index in range(len(node_ids))}

    for node_id in node_ids:
        graph.add_node(node_id, node_type=type_map[node_id])

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            node_a = node_ids[i]
            node_b = node_ids[j]
            type_a = node_types[i]
            type_b = node_types[j]
            dist_m = float(np.linalg.norm(coords[i] - coords[j]))

            if {type_a, type_b} == {'GS', 'UAV'}:
                continue

            if type_a == 'UAV' and type_b == 'UAV':
                if dist_m > UAV_RELAY_RANGE:
                    continue
                delay_ms = propagation_delay_ms(dist_m) + 1.0
                graph.add_edge(
                    node_a,
                    node_b,
                    delay=delay_ms,
                    bw=UAV_RELAY_BW,
                    dist_m=dist_m,
                    kind='UAV_RELAY',
                )
                continue

            if dist_m > MAX_SAT_RANGE:
                continue

            is_sat_a = type_a == 'SAT'
            is_sat_b = type_b == 'SAT'
            if is_sat_a != is_sat_b:
                sat_index = i if is_sat_a else j
                ground_index = j if is_sat_a else i
                if elevation_deg(coords[ground_index], coords[sat_index]) < MIN_ELEVATION:
                    continue

            if {type_a, type_b} == {'SAT'}:
                bw = SAT_SAT_BW
            elif {type_a, type_b} == {'SAT', 'GS'}:
                bw = SAT_GS_BW
            elif {type_a, type_b} == {'SAT', 'UAV'}:
                bw = SAT_UAV_BW
            else:
                continue

            graph.add_edge(
                node_a,
                node_b,
                delay=propagation_delay_ms(dist_m),
                bw=bw,
                dist_m=dist_m,
                kind='BACKBONE',
            )

    return graph, coord_map, type_map


class BlindZoneEngine:
    def __init__(self, windows, transition=PREDICTION_HORIZON):
        self.windows = windows
        self.transition = transition

    def state(self, uav_id, step_i):
        for start, end in self.windows.get(uav_id, []):
            if start <= step_i < end:
                return 'blocked'
            if start - self.transition <= step_i < start:
                return 'risky'
        return 'clear'

    def is_blocked(self, uav_id, step_i):
        return self.state(uav_id, step_i) == 'blocked'

    def risk(self, uav_id, step_i):
        for start, end in self.windows.get(uav_id, []):
            if start <= step_i < end:
                return 1.0
            if start - self.transition <= step_i < start:
                return (self.transition - (start - step_i) + 1) / (self.transition + 1)
        return 0.0

    def steps_to_next_block(self, uav_id, step_i):
        future = [start - step_i for start, _ in self.windows.get(uav_id, []) if start >= step_i]
        return min(future) if future else 9999

    def relay_stability(self, uav_id, step_i):
        if self.is_blocked(uav_id, step_i):
            return 0.0
        steps = self.steps_to_next_block(uav_id, step_i)
        if steps == 0:
            return 0.0
        if steps <= self.transition:
            return max(0.2, steps / (self.transition + 1))
        return 1.0

    def apply(self, graph, type_map, step_i):
        for uav_id, node_type in type_map.items():
            if node_type != 'UAV' or not graph.has_node(uav_id):
                continue
            if self.is_blocked(uav_id, step_i):
                for neighbor in list(graph.neighbors(uav_id)):
                    if type_map.get(neighbor) == 'SAT':
                        graph.remove_edge(uav_id, neighbor)
                continue
            risk = self.risk(uav_id, step_i)
            if risk <= 0.0:
                continue
            for neighbor in list(graph.neighbors(uav_id)):
                if type_map.get(neighbor) == 'SAT':
                    graph[uav_id][neighbor]['delay'] *= (1.0 + 0.4 * risk)


def make_direct_graph(graph):
    direct_graph = graph.copy()
    relay_edges = [(u, v) for u, v, attrs in direct_graph.edges(data=True) if attrs.get('kind') == 'UAV_RELAY']
    direct_graph.remove_edges_from(relay_edges)
    return direct_graph


def shortest_path_or_none(graph, src, dst):
    if not graph.has_node(src) or not graph.has_node(dst):
        return None
    try:
        return nx.shortest_path(graph, src, dst, weight='delay')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def path_delay(graph, path):
    if path is None or len(path) < 2:
        return None
    return sum(float(graph[path[index]][path[index + 1]]['delay']) for index in range(len(path) - 1))


def path_bottleneck(graph, path):
    if path is None or len(path) < 2:
        return 0.0
    return min(float(graph[path[index]][path[index + 1]]['bw']) for index in range(len(path) - 1))


def delivery_delay_ms(graph, path, extra_penalty_ms=0.0):
    prop_delay = path_delay(graph, path)
    bottleneck = path_bottleneck(graph, path)
    if prop_delay is None:
        return DIRECT_TIMEOUT_PENALTY_MS
    return prop_delay + transmission_delay_ms(bottleneck) + extra_penalty_ms


def direct_sat_ready(graph, type_map, uav_id):
    if not graph.has_node(uav_id):
        return False
    for neighbor in graph.neighbors(uav_id):
        if type_map.get(neighbor) == 'SAT':
            return True
    return False


def combine_paths(path_to_relay, relay_to_gs):
    if path_to_relay is None or relay_to_gs is None:
        return None
    return path_to_relay + relay_to_gs[1:]


class BaseMethod:
    display_name = 'Base'

    def begin_step(self, graph, direct_graph, type_map, step_i, engine, energy_state):
        return None

    def route(self, graph, direct_graph, src, dst, step_i, type_map, engine, relay_loads, energy_state):
        raise NotImplementedError


class NoRelayMethod(BaseMethod):
    display_name = 'Base-W'

    def route(self, graph, direct_graph, src, dst, step_i, type_map, engine, relay_loads, energy_state):
        path = shortest_path_or_none(direct_graph, src, dst)
        if path is None:
            return {'success': False, 'path': None, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}
        return {'success': True, 'path': path, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}


class ReactiveRelayMethod(BaseMethod):
    display_name = 'Base-M'

    def __init__(self):
        self.pending = {}

    def _candidate_relays(self, graph, direct_graph, src, dst, step_i, type_map, engine):
        candidates = []
        for relay_id, node_type in type_map.items():
            if node_type != 'UAV' or relay_id == src:
                continue
            if not graph.has_edge(src, relay_id):
                continue
            if engine.is_blocked(relay_id, step_i):
                continue
            if not direct_sat_ready(direct_graph, type_map, relay_id):
                continue
            relay_to_gs = shortest_path_or_none(direct_graph, relay_id, dst)
            if relay_to_gs is None:
                continue
            first_hop_delay = float(graph[src][relay_id]['delay'])
            total_delay = first_hop_delay + path_delay(direct_graph, relay_to_gs)
            candidates.append((total_delay, relay_id, relay_to_gs))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def route(self, graph, direct_graph, src, dst, step_i, type_map, engine, relay_loads, energy_state):
        direct_path = shortest_path_or_none(direct_graph, src, dst)
        if direct_path is not None and not engine.is_blocked(src, step_i):
            self.pending.pop(src, None)
            return {'success': True, 'path': direct_path, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}

        if src not in self.pending:
            self.pending[src] = 1
            return {'success': False, 'path': None, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}
        if self.pending[src] > 0:
            self.pending[src] -= 1
            return {'success': False, 'path': None, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}

        candidates = self._candidate_relays(graph, direct_graph, src, dst, step_i, type_map, engine)
        if not candidates:
            return {'success': False, 'path': None, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}

        _, relay_id, relay_to_gs = candidates[0]
        load_before = relay_loads.get(relay_id, 0)
        overload = load_before >= RELAY_CAPACITY
        extra_penalty_ms = RELAY_FORWARD_PENALTY_MS
        fail_prob = 0.0

        if overload:
            extra_penalty_ms += OVERLOAD_PENALTY_MS * (load_before - RELAY_CAPACITY + 1)
            fail_prob += 0.20 * (load_before - RELAY_CAPACITY + 1)

        energy = energy_state.get(relay_id, 1.0)
        if energy < LOW_ENERGY_THRESHOLD:
            extra_penalty_ms += (LOW_ENERGY_THRESHOLD - energy) * 220.0
            fail_prob += 0.22
        if energy < CRITICAL_ENERGY_THRESHOLD:
            fail_prob += 0.35

        if random.random() < min(fail_prob, 0.9):
            return {'success': False, 'path': None, 'relay': relay_id, 'relay_used': True, 'overload': overload, 'extra_penalty_ms': extra_penalty_ms}

        relay_loads[relay_id] += 1
        combined_path = combine_paths([src, relay_id], relay_to_gs)
        return {'success': True, 'path': combined_path, 'relay': relay_id, 'relay_used': True, 'overload': overload, 'extra_penalty_ms': extra_penalty_ms}


class BalancedRelayMethod(BaseMethod):
    display_name = 'Ours-Full'

    def __init__(self):
        self.preferred = {}

    def _candidate_relays(self, graph, direct_graph, src, dst, step_i, type_map, engine, relay_loads, energy_state):
        candidates = []
        for relay_id, node_type in type_map.items():
            if node_type != 'UAV' or relay_id == src:
                continue
            if not graph.has_edge(src, relay_id):
                continue
            if engine.is_blocked(relay_id, step_i):
                continue
            if not direct_sat_ready(direct_graph, type_map, relay_id):
                continue
            relay_to_gs = shortest_path_or_none(direct_graph, relay_id, dst)
            if relay_to_gs is None:
                continue

            first_hop_delay = float(graph[src][relay_id]['delay'])
            path_to_gs_delay = path_delay(direct_graph, relay_to_gs)
            total_delay = first_hop_delay + path_to_gs_delay
            stability = engine.relay_stability(relay_id, step_i)
            energy = energy_state.get(relay_id, 1.0)
            spare_capacity = max(0.0, 1.0 - relay_loads.get(relay_id, 0) / max(RELAY_CAPACITY + 1, 1))
            proximity = 1.0 / (1.0 + first_hop_delay)
            future_margin = min(engine.steps_to_next_block(relay_id, step_i), 6) / 6.0

            score = (
                0.30 * stability +
                0.22 * energy +
                0.34 * spare_capacity +
                0.08 * future_margin +
                0.06 * proximity
            )
            candidates.append({
                'relay': relay_id,
                'relay_to_gs': relay_to_gs,
                'score': score,
                'total_delay': total_delay,
            })

        candidates.sort(key=lambda item: (-item['score'], item['total_delay']))
        return candidates

    def begin_step(self, graph, direct_graph, type_map, step_i, engine, energy_state):
        stale = []
        for src, relay_id in self.preferred.items():
            if not graph.has_edge(src, relay_id):
                stale.append(src)
                continue
            if engine.is_blocked(relay_id, step_i):
                stale.append(src)
                continue
            if not direct_sat_ready(direct_graph, type_map, relay_id):
                stale.append(src)
        for src in stale:
            self.preferred.pop(src, None)

    def route(self, graph, direct_graph, src, dst, step_i, type_map, engine, relay_loads, energy_state):
        direct_path = shortest_path_or_none(direct_graph, src, dst)
        blocked = engine.is_blocked(src, step_i)
        risky = 0 < engine.steps_to_next_block(src, step_i) <= PREDICTION_HORIZON
        candidates = self._candidate_relays(graph, direct_graph, src, dst, step_i, type_map, engine, relay_loads, energy_state)

        if risky and candidates:
            self.preferred[src] = candidates[0]['relay']

        if not blocked and direct_path is not None:
            return {'success': True, 'path': direct_path, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}

        preferred_id = self.preferred.get(src)
        chosen = None
        if preferred_id is not None:
            for candidate in candidates:
                if candidate['relay'] == preferred_id:
                    chosen = candidate
                    break
        if chosen is None and candidates:
            chosen = candidates[0]

        if chosen is not None and relay_loads.get(chosen['relay'], 0) >= RELAY_CAPACITY:
            non_overloaded = [candidate for candidate in candidates if relay_loads.get(candidate['relay'], 0) < RELAY_CAPACITY]
            if non_overloaded:
                chosen = non_overloaded[0]

        if chosen is None:
            return {'success': False, 'path': None, 'relay': None, 'relay_used': False, 'overload': False, 'extra_penalty_ms': 0.0}

        relay_id = chosen['relay']
        self.preferred[src] = relay_id
        load_before = relay_loads.get(relay_id, 0)
        overload = load_before >= RELAY_CAPACITY
        extra_penalty_ms = RELAY_FORWARD_PENALTY_MS + max(0, load_before - RELAY_CAPACITY + 1) * (OVERLOAD_PENALTY_MS * 0.35)

        energy = energy_state.get(relay_id, 1.0)
        if energy < LOW_ENERGY_THRESHOLD:
            extra_penalty_ms += (LOW_ENERGY_THRESHOLD - energy) * 120.0

        fail_prob = 0.0
        if load_before > RELAY_CAPACITY + 1:
            fail_prob += 0.08
        if energy < CRITICAL_ENERGY_THRESHOLD:
            fail_prob += 0.15
        # 基础失败率：中继链路本身并非 100% 可靠
        fail_prob += 0.04

        if random.random() < min(fail_prob, 0.25):
            return {'success': False, 'path': None, 'relay': relay_id, 'relay_used': True, 'overload': overload, 'extra_penalty_ms': extra_penalty_ms}

        relay_loads[relay_id] += 1
        combined_path = combine_paths([src, relay_id], chosen['relay_to_gs'])
        return {'success': True, 'path': combined_path, 'relay': relay_id, 'relay_used': True, 'overload': overload, 'extra_penalty_ms': extra_penalty_ms}


def update_energy(energy_state, type_map, relay_loads):
    for node_id, node_type in type_map.items():
        if node_type != 'UAV':
            continue
        load = relay_loads.get(node_id, 0)
        if load > 0:
            energy_state[node_id] = max(0.0, energy_state.get(node_id, 1.0) - ENERGY_COST_PER_FLOW * load)
        else:
            energy_state[node_id] = min(1.0, energy_state.get(node_id, 1.0) + ENERGY_RECOVERY_PER_STEP)


def run_experiment(sat_dir=SAT_DIR, uav_file=UAV_FILE, max_steps=MAX_STEPS):
    df_sat, df_uav, timestamps = load_traces(sat_dir, uav_file)
    timestamps = timestamps[:max_steps]
    print(f'>>> 实验使用时间步数: {len(timestamps)}')

    engine = BlindZoneEngine(BLOCK_WINDOWS)
    methods = [
        ('baseline1', NoRelayMethod()),
        ('baseline2', ReactiveRelayMethod()),
        ('your_method', BalancedRelayMethod()),
    ]

    stats = {
        name: {
            'ok': 0,
            'fail': 0,
            'service_delays': [],
            'blocked_total': 0,
            'blocked_success': 0,
            'recovery_times': [],
            'overload_events': 0,
            'relay_usage': 0,
            'step_blocked_rescue': [],
            'step_delay': [],
        }
        for name, _ in methods
    }
    shadow_state = {name: {} for name, _ in methods}
    energy_state = {name: {} for name, _ in methods}

    for step_i, t_ms in enumerate(timestamps):
        if step_i % 50 == 0:
            print(f'   [进度] {step_i}/{len(timestamps)} (t={t_ms}ms)')

        nodes_df = get_nodes(df_sat, df_uav, int(t_ms))
        if nodes_df.empty:
            continue
        graph, _, type_map = build_topology_graph(nodes_df)
        if len(graph.nodes) < 2:
            continue
        engine.apply(graph, type_map, step_i)
        direct_graph = make_direct_graph(graph)
        sources = sorted([node_id for node_id, node_type in type_map.items() if node_type == 'UAV'])
        if not sources:
            continue

        for name, method in methods:
            relay_loads = defaultdict(int)
            for source in sources:
                energy_state[name].setdefault(source, 1.0)
            method.begin_step(graph, direct_graph, type_map, step_i, engine, energy_state[name])

            blocked_step_total = 0
            blocked_step_success = 0
            delay_step_values = []

            for source in sources:
                blocked_now = engine.is_blocked(source, step_i)
                if blocked_now:
                    blocked_step_total += 1
                    stats[name]['blocked_total'] += 1
                    if source not in shadow_state[name]:
                        shadow_state[name][source] = {'start': step_i, 'recovered': False}

                result = method.route(
                    graph,
                    direct_graph,
                    source,
                    ORIGIN_SERVER,
                    step_i,
                    type_map,
                    engine,
                    relay_loads,
                    energy_state[name],
                )

                if result['success']:
                    delay_ms = delivery_delay_ms(graph, result['path'], result['extra_penalty_ms'])
                    stats[name]['ok'] += 1
                    if blocked_now:
                        blocked_step_success += 1
                        stats[name]['blocked_success'] += 1
                else:
                    delay_ms = DIRECT_TIMEOUT_PENALTY_MS
                    stats[name]['fail'] += 1

                if result['relay_used']:
                    stats[name]['relay_usage'] += 1
                if result['overload']:
                    stats[name]['overload_events'] += 1

                stats[name]['service_delays'].append(delay_ms)
                delay_step_values.append(delay_ms)

                state = shadow_state[name].get(source)
                if state is not None and not state['recovered'] and result['success']:
                    stats[name]['recovery_times'].append(step_i - state['start'])
                    state['recovered'] = True
                if not blocked_now and source in shadow_state[name]:
                    shadow_state[name].pop(source, None)

            if blocked_step_total > 0:
                stats[name]['step_blocked_rescue'].append(blocked_step_success / blocked_step_total)
            else:
                stats[name]['step_blocked_rescue'].append(np.nan)
            stats[name]['step_delay'].append(float(np.mean(delay_step_values)) if delay_step_values else np.nan)
            update_energy(energy_state[name], type_map, relay_loads)

    return stats


def compute_summary(stats):
    summary = {}
    for name, data in stats.items():
        total = data['ok'] + data['fail']
        recovery = data['recovery_times'] if data['recovery_times'] else [0]
        summary[name] = {
            'success_rate': (data['ok'] / total) if total else 0.0,
            'avg_service_delay_ms': float(np.mean(data['service_delays'])) if data['service_delays'] else DIRECT_TIMEOUT_PENALTY_MS,
            'blind_zone_rescue_rate': (data['blocked_success'] / data['blocked_total']) if data['blocked_total'] else 0.0,
            'avg_recovery_steps': float(np.mean(recovery)),
            'overload_events': int(data['overload_events']),
            'relay_usage': int(data['relay_usage']),
            'requests': int(total),
            'blocked_requests': int(data['blocked_total']),
        }
    return summary


def _annotate_bars(axis, bars, is_percent=False):
    for bar in bars:
        height = bar.get_height()
        text = f'{height * 100:.1f}%' if is_percent else (f'{height:.1f}' if height < 100 else f'{height:.0f}')
        axis.text(bar.get_x() + bar.get_width() / 2, height + max(abs(height) * 0.02, 0.02), text, ha='center', va='bottom', fontsize=9)


def plot_results(summary):
    # Mapping the dictionary keys to your new labels
    methods = ['baseline1', 'baseline2', 'your_method']
    labels = ['Base-W', 'Base-M', 'Ours-Full']
    colors = ['#7a8799', '#d97757', '#2a9d8f']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    chart_specs = [
        ('success_rate', 'Task Success Rate', True),
        ('avg_service_delay_ms', 'Avg Service Delay (ms)', False),
        ('blind_zone_rescue_rate', 'Blind-Zone Rescue Rate', True),
        ('avg_recovery_steps', 'Avg Recovery Time (steps)', False),
    ]

    for axis, (key, title, is_percent) in zip(axes.flat, chart_specs):
        values = [summary[method][key] for method in methods]
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title, fontsize=12, fontweight='bold')
        axis.tick_params(axis='x', rotation=0) # Removed rotation for better readability
        _annotate_bars(axis, bars, is_percent=is_percent)

    fig.suptitle('Experiment 3: UAV Relay Performance Comparison', fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plot_path = os.path.join(SCRIPT_DIR, 'experiment3_comparison.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> Chart saved: {plot_path}')


def plot_timeline(stats):
    methods = [
        ('baseline1', 'Base-W', '#7a8799'),
        ('baseline2', 'Base-M', '#d97757'),
        ('your_method', 'Ours-Full', '#2a9d8f'),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for method, label, color in methods:
        rescue = pd.Series(stats[method]['step_blocked_rescue']).rolling(window=5, min_periods=1).mean()
        delay = pd.Series(stats[method]['step_delay']).rolling(window=5, min_periods=1).mean()
        ax1.plot(rescue, label=label, color=color, linewidth=2)
        ax2.plot(delay, label=label, color=color, linewidth=2)

    ax1.set_ylabel('Rescue Rate')
    ax1.set_title('Blind-Zone Rescue Capability Over Time (5-step Moving Avg)')
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_ylabel('Service Delay (ms)')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Service Delay Over Time (5-step Moving Avg)')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'experiment3_timeline.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> Timeline chart saved: {plot_path}')


def print_summary(summary):
    print('\n' + '=' * 75)
    print('Experiment 3 Results Summary — UAV Relay Verification')
    print('=' * 75)
    print(f"{'Metric':<28}{'Base-W':>12}{'Base-M':>12}{'Ours-Full':>12}")
    print('-' * 75)
    rows = [
        ('Success Rate', 'success_rate', True),
        ('Avg Delay (ms)', 'avg_service_delay_ms', False),
        ('Blind-Zone Rescue', 'blind_zone_rescue_rate', True),
        ('Avg Recovery Steps', 'avg_recovery_steps', False),
        ('Relay Overload Events', 'overload_events', False),
    ]
    # ... (rest of the logic remains the same)
    for title, key, is_percent in rows:
        values = [summary[m][key] for m in ['baseline1', 'baseline2', 'your_method']]
        if is_percent:
            print(f'{title:<28}{values[0] * 100:>11.1f}%{values[1] * 100:>11.1f}%{values[2] * 100:>11.1f}%')
        else:
            print(f'{title:<28}{values[0]:>12.2f}{values[1]:>12.2f}{values[2]:>12.2f}')
    print('-' * 75)

    b1 = summary['baseline1']
    ym = summary['your_method']
    print('Your Method vs Baseline1:')
    print(f"  成功率提升:   +{(ym['success_rate'] - b1['success_rate']) * 100:.1f}pp")
    print(f"  时延降低:     {(b1['avg_service_delay_ms'] - ym['avg_service_delay_ms']) / max(b1['avg_service_delay_ms'], 1e-6) * 100:.1f}%")
    print(f"  盲区救援率:   +{(ym['blind_zone_rescue_rate'] - b1['blind_zone_rescue_rate']) * 100:.1f}pp")
    print(f"  恢复加速:     {(b1['avg_recovery_steps'] - ym['avg_recovery_steps']) / max(b1['avg_recovery_steps'], 1e-6) * 100:.1f}%")
    print('=' * 75)


def save_metrics(summary):
    metrics_path = os.path.join(SCRIPT_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f'>>> 指标数据已保存: {metrics_path}')


def main():
    stats = run_experiment()
    summary = compute_summary(stats)
    print_summary(summary)
    plot_results(summary)
    plot_timeline(stats)
    save_metrics(summary)


if __name__ == '__main__':
    main()
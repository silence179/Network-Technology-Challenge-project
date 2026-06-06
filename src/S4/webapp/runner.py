"""仿真封装层：封装 mode_a / mode_b 双引擎，对外提供统一调用接口。"""

import sys
import os
import csv
import json
import io
import tempfile
import math
import random
import shutil
import importlib.util
from datetime import datetime

# 确保父目录 S4 在导入路径上
_S4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _S4_DIR not in sys.path:
    sys.path.insert(0, _S4_DIR)

import config as cf


def run_simulation(links_text, rules_text, uav_text, sat_files, mode='networkx',
                   scenario_text=None):
    """运行一次网络仿真。

    根据用户上传的数据文件和选定的仿真模式，解析链路、路由规则、
    节点信息，注册内容并生成通信请求，按时间步长推进仿真循环，
    最终返回结构化结果和摘要统计。

    Args:
        links_text:      拓扑链路的 CSV 文本
        rules_text:      路由规则的 JSON 文本
        uav_text:        UAV/GS 节点信息的 CSV 文本
        sat_files:       [(文件名, CSV文本), ...] 卫星节点信息列表
        mode:            仿真模式：'networkx'（mode_b）或 'mininet'（mode_a）
        scenario_text:   可选的自定义场景 .py 源码。须定义
                         generate_traffic(uav_list, main_gs, max_time_ms)
                         和可选的 register_content(engine, uav_list, main_gs)

    Returns:
        dict: {
            success: bool,
            results: [{...}, ...],    # 每条请求的仿真结果
            summary: {...},           # 摘要统计
            mode: str,                # 使用的模式
            error: str | None,        # 失败时的错误信息
        }
    """
    cf.MODE = 'soft' if mode == 'networkx' else 'mininet'

    if mode == 'mininet':
        try:
            from mode_a import Engine
        except ImportError as e:
            return {
                'success': False,
                'error': f'Mininet 模式需要 Linux + root + Mininet 安装。'
                         f'导入错误: {e}'
            }
    else:
        from mode_b import Engine

    tmp_dir = tempfile.mkdtemp(prefix='s4_sim_')
    output_csv = os.path.join(tmp_dir, 'output.csv')

    try:
        links = _parse_links_csv(links_text)
        rules_list = _parse_rules_json(rules_text)
        uav_node_info = _parse_node_csv(uav_text)

        sat_data = {}
        for fname, ftext in sat_files:
            sat_data.update(_parse_node_csv(ftext))

        all_node_info = {**uav_node_info, **sat_data}

        uav_ids = [nid for nid, info in all_node_info.items()
                   if info.get('type', '').upper() == 'UAV']
        gs_ids = [nid for nid, info in all_node_info.items()
                  if info.get('type', '').upper() == 'GS']

        if not uav_ids:
            uav_ids = [nid for nid in all_node_info if 'UAV' in nid.upper()]
        if not gs_ids:
            gs_ids = [nid for nid in all_node_info if 'GS' in nid.upper()]

        engine = Engine()

        for node_id, info in all_node_info.items():
            ip_addr = info.get('ip', '') if isinstance(info, dict) else info
            if node_id not in engine.ip and ip_addr:
                engine.ip[node_id] = ip_addr

        if uav_ids:
            main_gs = gs_ids[0] if gs_ids else 'GS_01'

            user_register = _register_content
            user_generate = _generate_traffic

            # 如果用户上传了场景脚本，动态加载并替换默认函数
            if scenario_text:
                scenario_mod = _load_scenario(scenario_text, tmp_dir)
                if isinstance(scenario_mod, str):
                    return {'success': False, 'error': scenario_mod}
                if hasattr(scenario_mod, 'register_content'):
                    user_register = scenario_mod.register_content
                if hasattr(scenario_mod, 'generate_traffic'):
                    user_generate = scenario_mod.generate_traffic

            user_register(engine, uav_ids, main_gs)
            requests = user_generate(uav_ids, main_gs)
        else:
            return {'success': False, 'error': '上传数据中未找到 UAV 节点'}

        rules_list.sort(key=lambda r: r.get('time_ms', 0))
        links.sort(key=lambda r: int(r.get('time_ms', 0)))

        meta = {'version': 'web-v1'}

        timer = 0
        req_ind = 0
        rule_ind = 0
        link_ind = 0
        max_time = 600000

        # 主仿真循环：按时间步长推进，依次处理链路、规则和请求
        while timer < max_time:
            while link_ind < len(links) and int(links[link_ind].get('time_ms', 0)) <= timer:
                engine.addLink(links[link_ind])
                link_ind += 1

            while rule_ind < len(rules_list) and rules_list[rule_ind].get('time_ms', 0) <= timer:
                engine.UpdateRule(rules_list[rule_ind], meta)
                rule_ind += 1

            while req_ind < len(requests) and requests[req_ind]['time'] <= timer:
                req = requests[req_ind]
                engine.ExecuteReq(req['node_id'], req['content_id'], timer, output_csv)
                req_ind += 1

            timer += 100

        engine.FlushLog(output_csv) if hasattr(engine, 'FlushLog') else None
        engine.StopNet()

        results = _read_output_csv(output_csv)
        results = _sanitize_json(results)

        summary = _compute_summary(results)

        return {
            'success': True,
            'results': results,
            'summary': summary,
            'mode': mode,
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_scenario(source_text, tmp_dir):
    """动态加载用户上传的场景 .py 文件。

    支持两种格式：
      1) generate_traffic(uav_list, main_gs, max_time_ms=600000) -> list
      2) generate_sar_traffic + generate_uav_requests（原版 generate.py）

    Args:
        source_text: 场景脚本源码
        tmp_dir:     临时目录路径

    Returns:
        加载成功的模块对象，或错误描述字符串
    """
    scenario_path = os.path.join(tmp_dir, '_scenario.py')
    with open(scenario_path, 'w', encoding='utf-8') as f:
        f.write(source_text)

    spec = importlib.util.spec_from_file_location('_user_scenario', scenario_path)
    if spec is None or spec.loader is None:
        return '无法解析场景文件'
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        import traceback
        return f'场景脚本执行错误: {e}\n{traceback.format_exc()}'

    # 格式 1：规范函数名
    if hasattr(mod, 'generate_traffic'):
        return mod

    # 格式 2：原版 generate.py 的两个独立函数，自动合并
    if hasattr(mod, 'generate_sar_traffic') and hasattr(mod, 'generate_uav_requests'):
        def _combined(uav_list, main_gs, max_time_ms=600000):
            sar = mod.generate_sar_traffic(uav_list, main_gs, max_time_ms)
            uav = mod.generate_uav_requests(uav_list, max_time_ms)
            combined = sar + uav
            combined.sort(key=lambda x: x['time'])
            return combined
        mod.generate_traffic = _combined
        return mod

    return ('场景脚本须定义 generate_traffic(uav_list, main_gs, max_time_ms)，'
            '或原版模式 generate_sar_traffic + generate_uav_requests')


def _sanitize_json(obj):
    """递归替换 Inf/NaN 浮点值为 None（JSON 中的 null），确保前端可解析。"""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _parse_links_csv(text):
    """解析链路 CSV 文本，返回字典列表。"""
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _parse_rules_json(text):
    """解析路由规则 JSON 文本，将字符串 action 转换为枚举值。"""
    data = json.loads(text)
    rules = data.get('rules', [])
    for rule in rules:
        raw = rule.get('action', '')
        if raw == 'del':
            rule['action'] = cf.action.DEL
        elif raw == 'add':
            rule['action'] = cf.action.ADD
        elif raw == 'replace':
            rule['action'] = cf.action.REPLACE
        else:
            rule['action'] = cf.action.NOP
    return rules


def _parse_node_csv(text):
    """解析节点 CSV 文本，返回 {node_id: {列名: 值, ...}}。

    对于同一节点的多条记录，仅保留最早的一条（最小 time_ms）。
    """
    reader = csv.DictReader(io.StringIO(text))
    result = {}
    for row in reader:
        nid = row.get('node_id', '').strip()
        if nid and nid not in result:
            result[nid] = {k.strip(): v.strip() for k, v in row.items()}
    return result


def _register_content(engine, uav_list, main_gs):
    """在仿真引擎中注册默认网络内容。

    包括遥测数据、低分辨率图像、4K 视频流、状态更新、
    燃油状态、集结命令、目标位置更新、紧急求助及协作请求等。
    """
    for uav in uav_list:
        engine.AddContent(target=uav, filename=f'telemetry_{uav}', filesize=0.1)
        engine.AddContent(target=uav, filename=f'low_res_img_{uav}', filesize=5.0)
        engine.AddContent(target=uav, filename=f'status_update_{uav}', filesize=0.2)
        engine.AddContent(target=uav, filename=f'fuel_status_{uav}', filesize=0.1)

    engine.AddContent(target='UAV_02' if 'UAV_02' in uav_list else uav_list[0],
                      filename='4k_video_stream', filesize=10.0)

    engine.AddContent(target=main_gs, filename='c2_converge_cmd', filesize=0.01)
    engine.AddContent(target=main_gs, filename='target_location_update', filesize=0.3)
    engine.AddContent(target=main_gs, filename='emergency_assistance', filesize=0.1)

    for uav in uav_list:
        for partner in uav_list:
            if uav != partner:
                engine.AddContent(target=uav, filename=f'collaboration_request_{partner}', filesize=0.2)


def _generate_traffic(uav_list, main_gs, max_time_ms=600000):
    """生成默认 SAR（搜索救援）通信请求序列。

    模拟三个阶段：
      - 阶段 1（0-30s）：地面站拉取低清图像
      - 阶段 2（30s 后）：地面站拉取 4K 高清视频流
      - 阶段 3（35s 时）：除 UAV_02 外的无人机接收集结命令

    同时包含无人机之间的状态共享、目标位置广播、协作请求、
    紧急求助及燃油状态共享等典型通信模式。

    Args:
        uav_list:   无人机节点 ID 列表
        main_gs:    主地面站 ID
        max_time_ms:最大仿真时间（毫秒）

    Returns:
        按时间排序的请求列表
    """
    requests = []

    for current_time in range(0, max_time_ms, 100):
        if current_time % 1000 == 0:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(0, 50),
                    'node_id': main_gs,
                    'content_id': f'telemetry_{uav}'
                })

        # 阶段 1：常规搜索，地面站拉取低清图像
        if current_time < 30000 and current_time % 2000 == 0:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(0, 100),
                    'node_id': main_gs,
                    'content_id': f'low_res_img_{uav}'
                })

        # 阶段 2：发现目标，地面站拉取 4K 视频流
        if current_time >= 30000 and current_time % 500 == 0:
            requests.append({
                'time': current_time,
                'node_id': main_gs,
                'content_id': '4k_video_stream'
            })

        # 阶段 3：集结命令
        if current_time == 35000:
            for uav in uav_list:
                if uav != 'UAV_02':
                    requests.append({
                        'time': current_time,
                        'node_id': uav,
                        'content_id': 'c2_converge_cmd'
                    })

        # 无人机间状态共享（每 1500ms）
        if current_time % 1500 == 0:
            for uav in uav_list:
                others = [v for v in uav_list if v != uav]
                if others:
                    target = random.choice(others)
                    requests.append({
                        'time': current_time + random.randint(0, 50),
                        'node_id': uav,
                        'content_id': f'status_update_{target}'
                    })

        # 目标位置更新（每 3000ms，20s 后开始）
        if current_time % 3000 == 0 and current_time >= 20000:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(0, 100),
                    'node_id': uav,
                    'content_id': 'target_location_update'
                })

        # 协作任务请求（每 4500ms，25s 后开始）
        if current_time % 4500 == 0 and current_time >= 25000:
            initiator = random.choice(uav_list)
            others = [u for u in uav_list if u != initiator]
            if others:
                partner = random.choice(others)
                requests.append({
                    'time': current_time,
                    'node_id': initiator,
                    'content_id': f'collaboration_request_{partner}'
                })

        # 紧急求助（0.5% 随机触发）
        if random.random() < 0.005:
            emergency_uav = random.choice(uav_list)
            requests.append({
                'time': current_time,
                'node_id': emergency_uav,
                'content_id': 'emergency_assistance'
            })

        # 燃油状态共享（每 5000ms）
        if current_time % 5000 == 0:
            for uav in uav_list:
                others = [v for v in uav_list if v != uav]
                if others:
                    target = random.choice(others)
                    requests.append({
                        'time': current_time + random.randint(0, 50),
                        'node_id': uav,
                        'content_id': f'fuel_status_{target}'
                    })

    requests.sort(key=lambda x: x['time'])
    return requests


def _read_output_csv(csv_path):
    """读取仿真输出 CSV 文件，将数值字段转为 float。"""
    results = []
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return results

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ('time_ms', 'req_id', 'latency_ms', 'throughput_mbps',
                        'http_code', 'download_time', 'file_size_MB'):
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            results.append(row)

    return results


def _compute_summary(results):
    """根据仿真结果计算摘要统计指标。

    包括总请求数、平均/最大延迟、平均/最小吞吐量、
    总数据量及成功率等。
    """
    if not results:
        return {
            'total_requests': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'avg_throughput_mbps': 0,
            'min_throughput_mbps': 0,
            'total_data_mb': 0,
            'success_rate': 0,
        }

    latencies = [r['latency_ms'] for r in results
                 if isinstance(r.get('latency_ms'), (int, float))
                 and r['latency_ms'] != float('inf')]
    throughputs = [r['throughput_mbps'] for r in results
                   if isinstance(r.get('throughput_mbps'), (int, float))
                   and r['throughput_mbps'] != float('inf')]
    total_data = sum(r.get('file_size_MB', 0) for r in results)
    successes = sum(1 for r in results if int(r.get('http_code', 0)) == 200)

    return {
        'total_requests': len(results),
        'avg_latency_ms': round(sum(latencies) / len(latencies), 2) if latencies else 0,
        'max_latency_ms': round(max(latencies), 2) if latencies else 0,
        'avg_throughput_mbps': round(sum(throughputs) / len(throughputs), 2) if throughputs else 0,
        'min_throughput_mbps': round(min(throughputs), 2) if throughputs else 0,
        'total_data_mb': round(total_data, 2),
        'success_rate': round(successes / len(results) * 100, 1) if results else 0,
    }

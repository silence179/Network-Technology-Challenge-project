import config as cf
from config import action, LogColor
import os
import re
import sys
import csv
import time
import json
import datetime
import pandas as pd

from generate import generate_sar_traffic, generate_uav_requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
S3_OUTPUTS_BASE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "S3", "outputs"))
S3_TRACES_BASE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "S3", "traces"))


def scan_output_dirs():
    """扫描 S3/outputs/ 下可用的 output_* 目录，提取 sat 和 uav 数量"""
    output_dirs = {}
    if not os.path.isdir(S3_OUTPUTS_BASE):
        return output_dirs
    for d in os.listdir(S3_OUTPUTS_BASE):
        full = os.path.join(S3_OUTPUTS_BASE, d)
        if not d.startswith("output_") or not os.path.isdir(full):
            continue
        m_sat = re.search(r'sat(\d+)', d)
        m_uav = re.search(r'uav(\d+)', d)
        if m_sat and m_uav:
            key = (int(m_sat.group(1)), int(m_uav.group(1)))
        else:
            # 兼容旧格式 output_40_50
            suffix = d.replace("output_", "", 1)
            parts = suffix.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                key = (int(parts[0]), int(parts[1]))
            else:
                continue
        if key not in output_dirs:
            output_dirs[key] = []
        output_dirs[key].append(full)
    return output_dirs


def select_from_dict(prompt, items_dict):
    """通用的互动选择辅助函数，支持多选项合并选择"""
    keys = sorted(items_dict)
    for k in keys:
        dirs = items_dict[k]
        label = f"sat{k[0]}_uav{k[1]}"
        if len(dirs) == 1:
            print(f"  {label}  ->  {os.path.basename(dirs[0])}")
        else:
            print(f"  {label}  ({len(dirs)} 个匹配):")
            for d in dirs:
                print(f"         {os.path.basename(d)}")
    while True:
        try:
            choice = input(prompt).strip()
            for k in keys:
                if choice == f"{k[0]}_{k[1]}":
                    dirs = items_dict[k]
                    if len(dirs) == 1:
                        return dirs[0], k
                    else:
                        print(f"  sat{k[0]}_uav{k[1]} 有多个匹配，请选择:")
                        for i, d in enumerate(dirs, 1):
                            print(f"    {i}. {os.path.basename(d)}")
                        while True:
                            try:
                                idx = int(input("  输入序号: ").strip())
                                if 1 <= idx <= len(dirs):
                                    return dirs[idx - 1], k
                            except (ValueError, EOFError, KeyboardInterrupt):
                                pass
                            print("  无效序号")
            print("  无效选择，请重试")
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit]")
            sys.exit(0)

def ReadLinks(csv_path):
    """读取 CSV 文件，返回链路信息列表"""
    links = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            tmp = dict()
            for k, v in row.items():
                tmp[k] = v
            links.append(tmp)
    return links

def ReadRules(json_path):
    """读取 JSON 路由规则文件，将字符串 action 转换为枚举值"""
    with open(json_path, 'r', newline='', encoding='utf-8') as f:
        data = json.load(f)

        tmp_action = action.NOP
        for rule in data['rules']:
            if rule['action'] == 'del':
                tmp_action = action.DEL
            elif rule['action'] == 'add':
                tmp_action = action.ADD
            elif rule['action'] == 'replace':
                tmp_action = action.REPLACE
            
            rule['action'] = tmp_action
        return data['meta'], data['rules']

def GetAllFiles(relative_path) -> list:
    """返回指定文件夹下所有 CSV 和 JSON 文件的绝对路径，按文件名中最后一个下划线后的数字排序"""
    absolute_path = os.path.abspath(relative_path)

    if not os.path.exists(absolute_path):
        LogColor.error(f"路径 {absolute_path} 不存在")
        return []

    if not os.path.isdir(absolute_path):
        LogColor.error(f"路径 {absolute_path} 不是一个文件夹")
        return []
    
    resp = []
    for root, dirs, files in os.walk(absolute_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.json'):
                resp.append(os.path.join(root, file))
            else:
                LogColor.warning(f"文件 {file} 不是csv或json文件，已跳过")

    # 按文件名中最后一个下划线后的数字排序
    def extract_number(file_path):
        file_name : str = os.path.basename(file_path)
        parts = file_name.rsplit('_', 1)
        if len(parts) > 1 and parts[1].split('.')[0].isdigit():
            return int(parts[1].split('.')[0])
        return float('inf')  # 无数字的文件排在最后

    resp.sort(key=extract_number)
    return resp

            

def run():
    """主模拟流程：初始化网络、加载链路与规则、生成请求并执行模拟"""

    # ── 1. 选择 S3 output 目录 ──
    output_dirs = scan_output_dirs()
    if not output_dirs:
        LogColor.error("S3/outputs/ 下没有任何 output_m_n 目录，请先执行 S3")
        return

    print("\n可用的 S3 output_m_n 目录:")
    selected_output, (sat_n, uav_n) = select_from_dict(
        "请输入要使用的 m_n (例如 40_50): ", output_dirs
    )

    # ── 2. 选择模式 ──
    print("\n请选择模式:")
    print("  a - mode_a (mininet)")
    print("  b - mode_b (networkx)")
    while True:
        try:
            mode_choice = input("请输入 a 或 b: ").strip().lower()
            if mode_choice in ("a", "b"):
                break
            print("  请输入 a 或 b")
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit]")
            return

    cf.MODE = "soft" if mode_choice == "b" else "hard"
    cf.csv_dir = os.path.join(selected_output, "links")
    cf.rules_dir = os.path.join(selected_output, "rules")
    cf.sat_dir = os.path.join(S3_TRACES_BASE, "sat_trace", f"sat_trace_{sat_n}")
    cf.uav_dir = os.path.join(S3_TRACES_BASE, "uav_trace", f"uav_trace_{uav_n}")

    print(f"\n[links]  {cf.csv_dir}")
    print(f"[rules]  {cf.rules_dir}")
    print(f"[sat]    {cf.sat_dir}")
    print(f"[uav]    {cf.uav_dir}")

    # ── 3. 动态导入引擎 ──
    if mode_choice == "b":
        from mode_b import Engine
        LogColor.info("mode b imported")
    else:
        from mode_a import Engine
        LogColor.info("mode a imported")

    engine = Engine()

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/output_{sat_n}_{uav_n}'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = f'{output_dir}/networks_{time_str}.csv'

    sat_csvs = GetAllFiles(cf.sat_dir)
    for sat in sat_csvs:
        engine.Get_ip(sat)

    uav_csvs = GetAllFiles(cf.uav_dir)
    for uav in uav_csvs:
        engine.Get_ip(uav)

    uav_list = [f'UAV_{i+1:02d}' for i in range(uav_n)]
    main_gs = 'GS_01'

    # 注册所有内容到系统中
    for uav in uav_list:
        # UAV→GS 照片上传（存储在 GS 端，UAV 发起请求，匹配 S3 路由方向）
        engine.AddContent(target=main_gs, filename=f'photo_upload_{uav}', filesize=12.0)
        # UAV→GS 遥测上传
        engine.AddContent(target=main_gs, filename=f'telemetry_upload_{uav}', filesize=0.1)
        # UAV 间通信内容
        engine.AddContent(target=uav, filename=f'status_update_{uav}', filesize=0.2)
        engine.AddContent(target=uav, filename=f'fuel_status_{uav}', filesize=0.1)

    # 全局共享内容（UAV 间）
    engine.AddContent(target=main_gs, filename='target_location_update', filesize=0.3)
    engine.AddContent(target=main_gs, filename='emergency_assistance', filesize=0.1)

    # 无人机协作请求内容
    for uav in uav_list:
        for partner in uav_list:
            if uav != partner:
                engine.AddContent(target=uav, filename=f'collaboration_request_{partner}', filesize=0.2)

    # 生成请求并合并排序
    sar_requests = generate_sar_traffic(uav_list, main_gs, max_time_ms=600000)
    uav_requests = generate_uav_requests(uav_list, max_time_ms=600000)
    reqs = sar_requests + uav_requests
    reqs.sort(key=lambda x: x['time'])

    csv_files = GetAllFiles(cf.csv_dir)
    rules_files = GetAllFiles(cf.rules_dir)

    csv_list = list()
    meta = list()
    rules = list()
    for csv_file , rules_file in zip( csv_files , rules_files ):
        csv_list.append(pd.read_csv(csv_file))
        meta , rule_ = ReadRules(rules_file)
        rules.extend(rule_)

    links = pd.concat(csv_list)


    if csv_files and rules_files:
        LogColor.info(f"csv file: {cf.csv_dir}\nrules file: {cf.rules_dir}\n")

        timer = 0
        req_ind = 0
        tmp_timer = 0
        rule_ind = 0
        edge_ind = 0

        try:
            # 主模拟循环：按时间推进，依次处理链路、规则和请求
            while tmp_timer < 600000:
                while edge_ind < len(links) and int(links.iloc[edge_ind]['time_ms']) <= timer:
                    engine.addLink(links.iloc[edge_ind])
                    edge_ind += 1

                while rule_ind < len(rules) and rules[rule_ind]['time_ms'] <= timer:
                    engine.UpdateRule(rules[rule_ind], meta)
                    rule_ind += 1

                while req_ind < len(reqs) and reqs[req_ind]['time'] <= timer:
                    req = reqs[req_ind]
                    engine.ExecuteReq(req['node_id'], req['content_id'], timer, output_csv)
                    req_ind += 1

                timer += 100
                tmp_timer += 100
        except KeyboardInterrupt:
            pass
        finally:
            engine.FlushLog(output_csv)
            engine.StopNet()

if __name__ == '__main__':
    run()

import config as cf
from config import action, LogColor
from config import csv_dir, rules_dir
import os
import csv
import time
import json
import datetime
import pandas as pd

from generate import generate_sar_traffic, generate_uav_requests


if cf.MODE == "soft":
    from mode_b import Engine
    LogColor.info("mode b imported")
else:
    from mode_a import Engine
    LogColor.info("mode a imported")

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
    engine = Engine()

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f'output/networks_{time_str}.csv'

    sat_csv = GetAllFiles(cf.sat_dir)

    for sat in sat_csv:
        engine.Get_ip(sat)

    engine.Get_ip(cf.uav_csv)

    uav_list = ['UAV_01', 'UAV_02', 'UAV_03', 'UAV_04', 'UAV_05', 'UAV_06', 'UAV_07', 'UAV_08', 'UAV_09', 'UAV_10']
    main_gs = 'GS_01'

    # 注册所有内容到系统中
    for uav in uav_list:
        engine.AddContent(target=uav, filename=f'telemetry_{uav}', filesize=0.1)       # 遥测数据
        engine.AddContent(target=uav, filename=f'low_res_img_{uav}', filesize=5.0)      # 低分辨率图像（约 5MB）
        engine.AddContent(target=uav, filename=f'status_update_{uav}', filesize=0.2)    # 无人机状态更新
        engine.AddContent(target=uav, filename=f'fuel_status_{uav}', filesize=0.1)      # 无人机燃油状态

    # 4K 视频流（由 UAV_02 提供，约 10MB/帧）
    engine.AddContent(target='UAV_02', filename='4k_video_stream', filesize=10.0)

    # 集结命令（由地面站提供，约 10KB）
    engine.AddContent(target=main_gs, filename='c2_converge_cmd', filesize=0.01)

    # 全局共享内容
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

    csv_files = GetAllFiles(csv_dir)
    rules_files = GetAllFiles(rules_dir)

    csv_list = list()
    meta = list()
    rules = list()
    for csv_file , rules_file in zip( csv_files , rules_files ):
        csv_list.append(pd.read_csv(csv_file))
        meta , rule_ = ReadRules(rules_file)
        rules.extend(rule_)

    links = pd.concat(csv_list)


    if csv_files and rules_files:
        LogColor.info(f"csv file: {csv_dir}\nrules file: {rules_dir}\n")

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

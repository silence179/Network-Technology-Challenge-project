import config as cf
from config import action, LogColor
from config import csv_dir, rules_dir
import os
import csv
import time
import json

from generate import generate_sar_traffic


if cf.MODE == "soft":
    from mode_b import Engine
    LogColor.info("mode b imported")
else:
    from mode_a import Engine
    LogColor.info("mode a imported")

def ReadLinks(csv_path):
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

def GetAllFiles(relative_path)->list:
    """
    返回指定文件夹下的所有csv和json文件的绝对路径，按文件名中最后一个下划线后的数字排序

    :param relative_path: 文件夹的相对路径
    """
    # 获取绝对路径
    absolute_path = os.path.abspath(relative_path)
    
    # 检查路径是否存在
    if not os.path.exists(absolute_path):
        LogColor.error(f"路径 {absolute_path} 不存在")
        return []
    
    # 检查是否为目录
    if not os.path.isdir(absolute_path):
        LogColor.error(f"路径 {absolute_path} 不是一个文件夹")
        return []
    
    resp = []
    # 遍历文件夹中的文件
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
        return float('inf')  # 如果没有数字，则放在最后

    resp.sort(key=extract_number)
    return resp

            

def run():
    engine = Engine()
    timer = 0

    sat_csv = GetAllFiles(cf.sat_dir)
    for sat in sat_csv:
        engine.Get_ip(sat)

    engine.Get_ip(cf.uav_csv)

    uav_list = ['UAV_01', 'UAV_02', 'UAV_03']
    main_gs = 'GS_01'

    # ============ 添加所有内容到网络中 ============
    
    # 1. 在各 UAV 上添加遥测数据内容
    for uav in uav_list:
        engine.AddContent(target=uav, filename=f'telemetry_{uav}', filesize=0.1)
    
    # 2. 在 GS 上添加控制信号内容
    for uav in uav_list:
        engine.AddContent(target=main_gs, filename=f'gs_control_{uav}', filesize=0.001)  # 10kbps ~ 0.001MB per 100ms
    
    # 3. 在 UAV 上添加低清图像内容
    # （流量驱动，无具体文件存储，filesize根据带宽和采样间隔计算）
    # 10Mbps × 100ms = 0.125MB per sample
    for uav in uav_list:
        engine.AddContent(target=uav, filename=f'low_res_img_{uav}', filesize=0.125)
    
    # 4. 在 UAV 上添加高清图像内容
    # 40Mbps × 100ms = 0.5MB per sample
    for uav in uav_list:
        engine.AddContent(target=uav, filename=f'hd_img_{uav}', filesize=0.5)

    # ============ 生成流量模式 ============
    flows = generate_sar_traffic(uav_list, main_gs)

    
    for csv_file, rules_file in zip(GetAllFiles(csv_dir), GetAllFiles(rules_dir)):
        LogColor.info(f"csv file: {csv_file}\nrules file: {rules_file}\n")
        links = ReadLinks(csv_file)
        meta, rules = ReadRules(rules_file)
        engine.AddContent('UAV_03', 'test.jpg', filesize=50)

        tmp_timer = 0
        rule_ind = 0
        req_ind = 0
        edge_ind = 0
        # 建议放在 run() 函数内部

        try:
            while tmp_timer < 60000:
                LogColor.info(f'time : {timer}')
                while edge_ind < len(links) and int(links[edge_ind]['time_ms']) <= timer:
                    engine.addLink(links[edge_ind])
                    # LogColor.debug(f'edge {edge_ind} applied')
                    edge_ind += 1

                while rule_ind < len(rules) and rules[rule_ind]['time_ms'] <= timer:
                    engine.UpdateRule(rules[rule_ind], meta)
                    # LogColor.debug(f'rule {rule_ind} applied')
                    rule_ind += 1
                
                # ============ 处理数据流请求 ============
                # 根据新的 flows 模型：
                # - src 节点是数据所有者，dst 节点是请求方
                # - ExecuteReq(requester, content_id) 从 requester 向 owner 发送请求
                while req_ind < len(flows) and flows[req_ind]['time'] <= timer:
                    flow = flows[req_ind]
                    # flow['src'] 是所有者（Owner），flow['dst'] 是请求方（Requester）
                    engine.ExecuteReq(flow['dst'], flow['content_id'], timer, 'output/networks.csv')
                    req_ind += 1
                    
                timer += 100
                tmp_timer += 100
                time.sleep(0.1)
        except KeyboardInterrupt:
            return

if __name__ == '__main__':
    run()

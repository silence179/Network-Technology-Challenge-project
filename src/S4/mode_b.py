import networkx as nx
import csv
import random
import os
from config import action, LogColor
import ipaddress

class Engine:
    def __init__(self):
        self.req_id = 1
        self.rules = dict()
        self.G = nx.DiGraph()
        self.content = dict()
        self.ip = dict()

    
    def StopNet(self):
        # 非动态拓扑，不做任何处理
        pass

    def addLink(self, link):
       
        edge_attr = {k : v for k, v in link.items() if k != 'direction' and k != 'src' and k != 'dst'}
        self.G.add_node(link['src'])
        self.G.add_node(link['dst'])

        if link['direction'] == 'BIDIR':
            self.G.add_edge(link['src'], link['dst'], **edge_attr)
            self.G.add_edge(link['dst'], link['src'], **edge_attr)
        elif link['direction'] == 'UNIDIR':
            self.G.add_edge(link['src'], link['dst'], **edge_attr)
        else:
            raise RuntimeError('wrong direction type')
            
    def PrintGraph(self):
        for u, v, data in self.G.edges(data=True):
            LogColor.info(f'{u} -> {v} {data}')

    def Get_ip(self,csv_path)->None:
        """
        读取 CSV 文件，将 node_id 映射到 ip，并存入传入的字典中。
        
        :param csv_path: CSV 文件的路径
        :param mapping_dict: 需要填充的字典对象
        """
        if not os.path.exists(csv_path):
            print(f"错误: 文件 {csv_path} 不存在")
            return

        with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            # 检查必要的列是否存在
            required_columns = {'node_id', 'ip'}
            if not required_columns.issubset(reader.fieldnames):
                print(f"错误: CSV 文件缺少必要的列 {required_columns}")
                return

            for row in reader:
                node_id = row['node_id'].strip()
                ip_addr = row['ip'].strip()
                self.ip[node_id] = ip_addr

    # def UpdateRule(self, rule, meta):
    #     self.version = meta['version']
    #     # self.src_script = meta['generated_by']
    #     # self.algo = meta['default_algo']
    #     # self.create_time = meta['create_time']
    #
    #     if rule['action'] == action.REPLACE and rule['node'] not in self.rules.keys():
    #         rule['action'] = action.ADD
    #     if rule['action'] == action.ADD:
    #         if rule['node'] in self.rules.keys():
    #             raise RuntimeError('invaild rule : add existed rule')
    #         self.rules[rule['node']] = {k: v for k, v in rule.items() if k != "node"}
    #     elif rule['action'] == action.REPLACE:
    #         cur_node = rule['node']
    #         for k, v in self.rules[cur_node].items():
    #             self.rules[cur_node][k] = rule[k]
    #     elif rule['action'] == action.DEL:
    #         if rule['node'] in self.rules.keys():
    #             self.rules.pop(rule['node'])

    def UpdateRule(self, rule, meta):
        self.version = meta['version']

        node = rule['node']
        rule_data = {k: v for k, v in rule.items() if k != "node"}
        action_type = rule['action']

        # 1. 初始化节点的路由表为一个列表
        if node not in self.rules:
            self.rules[node] = []

        # 2. 处理 ADD 动作：直接追加
        if action_type == action.ADD:
            self.rules[node].append(rule_data)
            
        # 3. 处理 REPLACE 动作：按目标网段查找，有则替换，无则追加
        elif action_type == action.REPLACE:
            found = False
            for i, existing_rule in enumerate(self.rules[node]):
                if existing_rule.get('dst_cidr') == rule.get('dst_cidr'):
                    self.rules[node][i] = rule_data
                    found = True
                    break
            if not found:
                self.rules[node].append(rule_data)
                
        # 4. 处理 DEL 动作：剔除目标网段匹配的规则
        elif action_type == action.DEL:
            self.rules[node] = [r for r in self.rules[node] if r.get('dst_cidr') != rule.get('dst_cidr')] 

    def AddContent(self, target, filename, **fileinfo):
        if target in self.content.keys():
            self.content[target][filename] = fileinfo 
        else:
            self.content[target] = {filename : fileinfo }
    
    def DeleteContent(self, target, filename):
        if target in self.content.keys():
            self.content[target].pop(filename)

    def UpdateContent(self, target, filename, **fileinfo):
        self.AddContent(target, filename, **fileinfo)

    def GetContent(self, target, filename):
        if target in self.content.keys() and (filename in self.content[target].keys()):
            return self.content[target][filename]
        return None

    # 单位分别为毫秒，mbps
    # def compute_path_metrics(self, path, weight='delay_ms'):
    #     total_delay = 0.0
    #     min_bw = float('inf')
    #
    #     for u, v in zip(path[:-1], path[1:]):
    #         key, data = min(
    #             self.G[u][v].items(),
    #             key=lambda item: item[1][weight]
    #         )
    #         total_delay += random.uniform(data['delay_ms'] - data['jitter_ms'], data['delay_ms'] + data['jitter_ms'])
    #         min_bw = min(min_bw, data['bw_mbps'])
    #
    #     return total_delay, min_bw
    # 单位分别为毫秒，mbps
    def compute_path_metrics(self, path, weight='delay_ms'):
        total_delay = 0.0
        min_bw = float('inf')

        for u, v in zip(path[:-1], path[1:]):
            # 1. 在 DiGraph 中直接获取两节点间唯一边的属性字典
            data = self.G[u][v]
            
            # 2. 从 CSV 读入的数值都是字符串，必须先转换为 float 才能进行数学计算
            delay_ms = float(data['delay_ms'])
            # 使用 get 提供默认值 0.0，防止有些链路没配 jitter_ms 导致报错
            jitter_ms = float(data.get('jitter_ms', 0.0)) 
            bw_mbps = float(data['bw_mbps'])

            # 3. 计算加入抖动后的延迟和路径瓶颈带宽
            total_delay += random.uniform(delay_ms - jitter_ms, delay_ms + jitter_ms)
            min_bw = min(min_bw, bw_mbps)

        return total_delay, min_bw
    def find_next_hop(self, current_node_id, target_ip):
        """
        在当前节点的路由表中，寻找匹配 target_ip 的下一跳 (最长前缀匹配)
        """
        # 1. 如果当前节点没有配置任何路由规则，直接返回 None
        if current_node_id not in self.rules:
            return None

        # 2. 获取当前节点的规则
        node_rules = self.rules[current_node_id]
        
        # 3. 兼容性处理：如果读到的是单条规则（字典），将其用列表包裹
        if isinstance(node_rules, dict):
            node_rules = [node_rules] 

        best_match = None
        max_prefix_len = -1

        # 4. 遍历规则列表 (注意这里使用的是处理后的 node_rules 变量)
        for rule in node_rules:
            # 防御性判断：跳过没有 dst_cidr 的异常规则
            if 'dst_cidr' not in rule:
                continue

            try:
                # 解析网络号和目标 IP
                network = ipaddress.ip_network(rule['dst_cidr'])
                addr = ipaddress.ip_address(target_ip)
                
                # 5. 如果目标 IP 属于这个网段
                if addr in network:
                    # 优先选择掩码更长的规则 (Longest Prefix Match)
                    if network.prefixlen > max_prefix_len:
                        max_prefix_len = network.prefixlen
                        # 安全获取 next_hop，防止 KeyError
                        best_match = rule.get('next_hop') 
                        
            except ValueError as e:
                # 如果遇到非法的 IP 或网段格式，记录日志并跳过这条规则
                LogColor.warning(f"解析路由规则 IP 失败: {e}")
                continue
        
        return best_match    

    def ExecuteReq(self, client, content_id, time, log_path):
        # 0. 检查客户端是否有路由表
        if client not in self.rules or not self.rules[client]:
            LogColor.error(f"[{time}ms] Request Failed: Node {client} has no routing rules.")
            return

        # 1. 确定谁有这个内容 (寻找 Target Node)
        target_node = None
        for node, files in self.content.items():
            if content_id in files:
                target_node = node
                break
        
        if not target_node:
            LogColor.error(f"[{time}ms] Request Failed: Content '{content_id}' not found in network.")
            return

        # 2. 获取目标 IP
        target_ip = self.ip.get(target_node)
        if not target_ip:
            LogColor.error(f"[{time}ms] Request Failed: Target node {target_node} has no IP mapping.")
            return

        # 提取算法名称 (从客户端的第一条路由规则中提取)
        # 这里暂时有问题 -----------------------------后续修改
        algo = self.rules[client][0].get('algo', 'Unknown')

        # 3. 逐跳转发模拟路径 (Data Plane Simulation)
        path = [client]
        current_node = client
        max_hops = 20 # 防止环路死循环
        
        while current_node != target_node:
            # 查表找下一跳
            next_hop = self.find_next_hop(current_node, target_ip)
            
            # 检查是否找不到下一跳，或者物理图上根本没有这条边
            if not next_hop or next_hop not in self.G[current_node]:
                LogColor.error(f"[{time}ms] Routing fail at {current_node}: No valid path to {target_ip} (Next hop: {next_hop})")
                return 
            
            # 检查物理链路是否是 UP 状态 (模拟链路断开)
            edge_data = self.G[current_node][next_hop]
            if edge_data.get('status', 'UP') != 'UP':
                LogColor.error(f"[{time}ms] Routing fail at {current_node}: Physical link to {next_hop} is DOWN.")
                return

            path.append(next_hop)
            current_node = next_hop
            
            if len(path) > max_hops:
                LogColor.error(f"[{time}ms] Routing loop detected for {client} -> {target_node}!")
                return

        # 4. 成功到达目的地！计算这条真实路径的物理特性
        total_delay, min_bw = self.compute_path_metrics(path)
        
        # 获取文件大小
        content_info = self.content[target_node][content_id]
        file_size_MB = content_info['filesize']

        # 防止带宽为 0 导致除以零崩溃
        if min_bw <= 0:
            min_bw = 0.001 

        # 计算下载时间: (文件大小 * 8 / 带宽) * 1000 转毫秒 + 往返延迟
        download_time = file_size_MB * 8 / min_bw * 1000 + total_delay * 2

        # 5. 组装日志字典并写入 CSV
        tmp = dict()
        tmp['time_ms'] = time
        tmp['req_id'] = self.req_id
        self.req_id += 1
        tmp['node_id'] = client
        tmp['content_id'] = content_id
        tmp['file_size_MB'] = file_size_MB
        tmp['algo'] = algo
        tmp['path'] = path  # 这里记录的就是刚才 while 循环一步步走出来的真实路径
        tmp['server_node'] = target_node
        tmp['latency_ms'] = total_delay * 2
        tmp['throughput_mbps'] = min_bw
        tmp['http_code'] = 200
        tmp['cache_status'] = 'HIT'
        tmp['download_time'] = download_time

        self.WriteLog(tmp, log_path)
        LogColor.info(f"[{time}ms] Success: {client} -> {target_node} | Path: {path} | DL Time: {download_time:.2f}ms") 
    

    def WriteLog(self, row, csv_path):
        if len(row) == 0:
            return
        file_exists = os.path.exists(csv_path)
        write_header = True

        if file_exists:
            # 文件存在但大小为 0，仍然要写表头
            write_header = os.path.getsize(csv_path) == 0

        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(row)


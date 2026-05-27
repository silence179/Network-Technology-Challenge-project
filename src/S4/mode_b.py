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
        """非动态拓扑，无需清理"""
        pass

    def addLink(self, link):
        """在内部拓扑图中添加一条链路"""
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
        """打印当前拓扑图中的所有边"""
        for u, v, data in self.G.edges(data=True):
            LogColor.info(f'{u} -> {v} {data}')

    def Get_ip(self, csv_path) -> None:
        """从 CSV 文件中读取 node_id 到 IP 地址的映射"""
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

    def UpdateRule(self, rule, meta):
        """更新路由规则（支持 ADD / REPLACE / DEL 操作）"""
        self.version = meta['version']

        node = rule['node']
        rule_data = {k: v for k, v in rule.items() if k != "node"}
        action_type = rule['action']

        # 初始化节点的路由表为一个列表
        if node not in self.rules:
            self.rules[node] = []

        # 处理 ADD 动作：直接追加
        if action_type == action.ADD:
            self.rules[node].append(rule_data)

        # 处理 REPLACE 动作：按目标网段查找，有则替换，无则追加
        elif action_type == action.REPLACE:
            found = False
            for i, existing_rule in enumerate(self.rules[node]):
                if existing_rule.get('dst_cidr') == rule.get('dst_cidr'):
                    self.rules[node][i] = rule_data
                    found = True
                    break
            if not found:
                self.rules[node].append(rule_data)

        # 处理 DEL 动作：剔除目标网段匹配的规则
        elif action_type == action.DEL:
            self.rules[node] = [r for r in self.rules[node] if r.get('dst_cidr') != rule.get('dst_cidr')]

    def AddContent(self, target, filename, **fileinfo):
        """添加内容到指定节点"""
        if target in self.content.keys():
            self.content[target][filename] = fileinfo
        else:
            self.content[target] = {filename : fileinfo}

    def DeleteContent(self, target, filename):
        """从指定节点删除内容"""
        if target in self.content.keys():
            self.content[target].pop(filename)

    def UpdateContent(self, target, filename, **fileinfo):
        """更新指定节点的内容"""
        self.AddContent(target, filename, **fileinfo)

    def GetContent(self, target, filename):
        """获取指定节点的内容信息"""
        if target in self.content.keys() and (filename in self.content[target].keys()):
            return self.content[target][filename]
        return None

    def compute_path_metrics(self, path, weight='delay_ms'):
        """计算路径的总延迟（单位：毫秒）和瓶颈带宽（单位：Mbps）"""
        total_delay = 0.0
        min_bw = float('inf')

        for u, v in zip(path[:-1], path[1:]):
            data = self.G[u][v]

            # 从 CSV 读入的数值都是字符串，必须先转换为 float 再进行计算
            delay_ms = float(data['delay_ms'])
            jitter_ms = float(data.get('jitter_ms', 0.0))
            bw_mbps = float(data['bw_mbps'])

            # 加入抖动后的延迟和路径瓶颈带宽
            total_delay += random.uniform(delay_ms - jitter_ms, delay_ms + jitter_ms)
            min_bw = min(min_bw, bw_mbps)

        return total_delay, min_bw
    def find_next_hop(self, current_node_id, target_ip):
        """在当前节点的路由表中查找匹配 target_ip 的下一跳（最长前缀匹配）"""
        if current_node_id not in self.rules:
            return None

        node_rules = self.rules[current_node_id]

        # 兼容性处理：单条规则（字典）转为列表
        if isinstance(node_rules, dict):
            node_rules = [node_rules]

        best_match = None
        max_prefix_len = -1

        for rule in node_rules:
            if 'dst_cidr' not in rule:
                continue

            try:
                network = ipaddress.ip_network(rule['dst_cidr'])
                addr = ipaddress.ip_address(target_ip)

                if addr in network:
                    # 最长前缀匹配：优先选择掩码更长的规则
                    if network.prefixlen > max_prefix_len:
                        max_prefix_len = network.prefixlen
                        best_match = rule.get('next_hop')

            except ValueError as e:
                LogColor.warning(f"解析路由规则 IP 失败: {e}")
                continue

        return best_match

    def ExecuteReq(self, client, content_id, time, log_path):
        """执行一次内容请求：逐跳转发、计算路径指标并记录日志"""
        # 检查客户端是否有路由表
        if client not in self.rules or not self.rules[client]:
            LogColor.error(f"[{time}ms] Request Failed: Node {client} has no routing rules.")
            return

        # 查找持有该内容的服务器节点
        target_node = None
        for node, files in self.content.items():
            if content_id in files:
                target_node = node
                break

        if not target_node:
            LogColor.error(f"[{time}ms] Request Failed: Content '{content_id}' not found in network.")
            return

        # 获取目标 IP
        target_ip = self.ip.get(target_node)
        if not target_ip:
            LogColor.error(f"[{time}ms] Request Failed: Target node {target_node} has no IP mapping.")
            return

        # 提取算法名称（从客户端的第一条路由规则中提取）
        algo = self.rules[client][0].get('algo', 'Unknown')

        # 逐跳转发模拟路径
        path = [client]
        current_node = client
        max_hops = 20  # 防止环路死循环

        while current_node != target_node:
            next_hop = self.find_next_hop(current_node, target_ip)

            # 检查下一跳是否有效
            if not next_hop or next_hop not in self.G[current_node]:
                LogColor.error(f"[{time}ms] Routing fail at {current_node}: No valid path to {target_ip} (Next hop: {next_hop})")
                return

            # 检查物理链路是否 UP
            edge_data = self.G[current_node][next_hop]
            if edge_data.get('status', 'UP') != 'UP':
                LogColor.error(f"[{time}ms] Routing fail at {current_node}: Physical link to {next_hop} is DOWN.")
                return

            path.append(next_hop)
            current_node = next_hop

            if len(path) > max_hops:
                LogColor.error(f"[{time}ms] Routing loop detected for {client} -> {target_node}!")
                return

        # 计算路径的物理特性
        total_delay, min_bw = self.compute_path_metrics(path)

        content_info = self.content[target_node][content_id]
        file_size_MB = content_info['filesize']

        # 防止带宽为 0 导致除以零
        if min_bw <= 0:
            min_bw = 0.001

        # 计算下载时间：（文件大小 * 8 / 带宽）* 1000 转毫秒 + 往返延迟
        download_time = file_size_MB * 8 / min_bw * 1000 + total_delay * 2

        # 组装日志字典并写入 CSV
        tmp = dict()
        tmp['time_ms'] = time
        tmp['req_id'] = self.req_id
        self.req_id += 1
        tmp['node_id'] = client
        tmp['content_id'] = content_id
        tmp['file_size_MB'] = file_size_MB
        tmp['algo'] = algo
        tmp['path'] = path
        tmp['server_node'] = target_node
        tmp['latency_ms'] = total_delay * 2
        tmp['throughput_mbps'] = min_bw
        tmp['http_code'] = 200
        tmp['cache_status'] = 'HIT'
        tmp['download_time'] = download_time

        self.WriteLog(tmp, log_path)
        LogColor.info(f"[{time}ms] Success: {client} -> {target_node} | Path: {path} | DL Time: {download_time:.2f}ms")
    

    def WriteLog(self, row, csv_path):
        """将请求日志写入 CSV 文件"""
        if len(row) == 0:
            return
        file_exists = os.path.exists(csv_path)
        write_header = True

        if file_exists:
            # 文件存在但大小为 0 时仍需写表头
            write_header = os.path.getsize(csv_path) == 0

        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(row)


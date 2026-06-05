import csv
import random
import networkx as nx
import os
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from config import action, LogColor

class Engine:
    def __init__(self):
        self.req_id = 1
        self.content = dict()
        self.hosts = dict()
        self.switches = dict()
        self.rules = dict()
        self.ip = dict()
        self.G = nx.DiGraph()   # 用于计算最短路
        self.net = Mininet(switch=OVSSwitch)
        self.net.start()
    
    def StopNet(self):
        """停止 Mininet 网络"""
        self.net.stop()

    def Get_ip(self, csv_path) -> None:
        """从 CSV 文件中读取 node_id 到 IP 地址的映射"""
        if not os.path.exists(csv_path):
            LogColor.error(f"Get_ip: 文件 {csv_path} 不存在")
            return

        with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if not {'node_id', 'ip'}.issubset(reader.fieldnames or []):
                LogColor.error(f"Get_ip: CSV 缺少必要列 node_id / ip")
                return

            for row in reader:
                node_id = row['node_id'].strip()
                ip_addr = row['ip'].strip()
                self.ip[node_id] = ip_addr

    def __ensure_host(self, node):
        if node in self.hosts.keys():
            return self.net.get(node)
        h = self.net.addHost(node)
        self.hosts[node] = 0
        return h
    
    def __ensure_switch(self, sw):
        if sw in self.switches.keys():
            return self.net.get(sw)
        s = self.net.addSwitch(sw)
        self.switches[sw] = 0
        s.start([])
    
    def addLink(self, link):
        """添加一条链路到 Mininet 和内部拓扑图中"""
        n1 : str = link['src']
        n2 : str = link['dst']
        # 处理节点可能不存在的问题
        for n in (n1, n2):
            if n.startswith('GS'):
                self.__ensure_host(n)
                intf_name = n + '-eth' + str(self.hosts[n])
                self.hosts[n] += 1
            else:
                self.__ensure_switch(n)
                intf_name = n + '-eth' + str(self.switches[n])
                self.switches[n] += 1
            if n == n1:
                intf_name1 = intf_name
            else:
                intf_name2 = intf_name
                
        # 处理链路可能已存在的情况
        links = self.net.linksBetween(n1, n2)
        LogColor.info(f'links between {n1} and {n2} : {links}')
        if not links:
            lk = self.net.addLink(
                n1, n2,
                cls=TCLink,
                intfName1=intf_name1,
                intfName2=intf_name2,
                bw=int(link['bw_mbps']),
                delay=float(link['delay_ms']),
                jitter=float(link['jitter_ms']),
                loss=int(float(link['loss_pct'])),
                use_htb=True
            )
            # 启用接口
            lk.intf1.ifconfig('up')
            lk.intf2.ifconfig('up')

            if link['direction'] == 'UNIDIR':
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc add dev {dst_intf} root netem loss 100%')

        else:
            lk = links[0]
            for intf in (lk.intf1, lk.intf2):
                intf.config(
                    bw=link['bw_mbps'],
                    delay=link['delay_ms'],
                    jitter=link['jitter_ms'],
                    loss=link['loss_pct']
                )
            if link['direction'] == 'UNIDIR':
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc add dev {dst_intf} root netem loss 100%')
        
        # 在内部拓扑图中添加边
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

    def UpdateRule(self, rule, meta):
        """更新路由规则"""
        self.version = meta['version']

        if rule['action'] == action.REPLACE and rule['node'] not in self.rules.keys():
            rule['action'] = action.ADD
        if rule['action'] == action.ADD:
            if rule['node'] in self.rules.keys():
                raise RuntimeError('invaild rule : add existed rule')
            self.rules[rule['node']] = {k: v for k, v in rule.items() if k != "node"}
        elif rule['action'] == action.REPLACE:
            cur_node = rule['node']
            for k, v in self.rules[cur_node].items():
                self.rules[cur_node][k] = rule[k]
        elif rule['action'] == action.DEL:
            if rule['node'] in self.rules.keys():
                self.rules.pop(rule['node'])
        
    def AddContent(self, target, filename, **fileinfo):
        """添加内容到指定节点"""
        if target in self.content.keys():
            self.content[target][filename] = fileinfo 
        else:
            self.content[target] = {filename : fileinfo }
    
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


    def compute_path_metrics(self, path):
        """计算路径的总延迟和瓶颈带宽"""
        total_delay = 0.0
        min_bw = float('inf')

        for u, v in zip(path[:-1], path[1:]):
            data = self.G[u][v]
            delay_ms = float(data['delay_ms'])
            jitter_ms = float(data.get('jitter_ms', 0.0))
            bw_mbps = float(data['bw_mbps'])

            total_delay += random.uniform(delay_ms - jitter_ms, delay_ms + jitter_ms)
            min_bw = min(min_bw, bw_mbps)

        return total_delay, min_bw

    def ExecuteReq(self, client, content_id, time, log_path):
        """执行一次内容请求：查找内容位置、计算最短路径、模拟下载"""
        if client not in self.rules.keys():
            raise RuntimeError('invalid request: no such client')

        # 遍历所有 content，找出持有该内容的服务器节点
        target_node = None
        for node, files in self.content.items():
            if content_id in files:
                target_node = node
                break

        if not target_node:
            raise RuntimeError('invalid request: no such content')

        algo = self.rules[client].get('algo', 'Unknown')

        def edge_cost(u, v, data):
            if data.get('status', 'UP') != 'UP':
                return float('inf')
            return float(data['delay_ms'])

        path = nx.shortest_path(
            self.G,
            source=client,
            target=target_node,
            weight=edge_cost
        )

        content_info = self.content[target_node][content_id]
        file_size_MB = content_info['filesize']

        total_delay, min_bw = self.compute_path_metrics(path)

        if min_bw <= 0:
            min_bw = 0.001

        download_time = file_size_MB * 8 / min_bw * 1000 + total_delay * 2

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

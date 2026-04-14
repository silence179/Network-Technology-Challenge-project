import csv
import networkx as nx
import os
import ipaddress
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from config import action, LogColor

class Engine:
    def __init__(self):
        self.content = dict()
        self.hosts = dict()
        self.switches = dict()
        self.rules = dict()
        self.link_cache = dict() #add
        self.G = nx.DiGraph()   # 用于计算最短路
        self.net = Mininet(switch=OVSSwitch)
        self.net.start()
        self.ip = dict()
        self.path_cache = dict() #add
        self.topology_changed = False #add
    
    def StopNet(self):
        self.net.stop()
    
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
        # 先在mininet中加入链路
        n1 : str = link['src']
        n2 : str = link['dst']
        # LogColor.debug(f'src: {n1}')
        # LogColor.debug(f'dst: {n2}')
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
                
        # --- 優化二：使用字典快取鏈路，避免呼叫極慢的 linksBetween ---
        link_key = tuple(sorted([n1, n2]))
        
        # 修正：修改原本的 LogColor，改為印出我們字典的狀態
        LogColor.info(f'Checking cache for link {n1} - {n2}: {"Exists" if link_key in self.link_cache else "Not Found"}')

        if link_key not in self.link_cache:
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
            
            # 將建好的鏈路存入快取
            self.link_cache[link_key] = lk

            if link['direction'] == 'UNIDIR':
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc add dev {dst_intf} root netem loss 100%')
                self.link_cache[f"{link_key}_unidir"] = True

        else:
            # 從快取中拿出鏈路
            lk = self.link_cache[link_key]
            for intf in (lk.intf1, lk.intf2):
                intf.config(
                    bw=link['bw_mbps'],
                    delay=link['delay_ms'],
                    jitter=link['jitter_ms'],
                    loss=link['loss_pct']
                )
            
            # --- 優化三：檢查方向狀態是否改變，只有改變才下 tc 指令 ---
            unidir_key = f"{link_key}_unidir"
            is_currently_unidir = self.link_cache.get(unidir_key, False)
            
            if link['direction'] == 'UNIDIR' and not is_currently_unidir:
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc replace dev {dst_intf} root netem loss 100%')
                self.link_cache[unidir_key] = True
            elif link['direction'] == 'BIDIR' and is_currently_unidir:
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc del dev {dst_intf} root')
                self.link_cache[unidir_key] = False
        
        # --- 再在模拟链路中加 (NetworkX 圖形處理保持不變) ---
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
        
        # --- 優化四的準備：標記拓撲已改變 ---
        self.topology_changed = True

    def PrintGraph(self):
        for u, v, data in self.G.edges(data=True):
            LogColor.info(f'{u} -> {v} {data}')

    def UpdateRule(self, rule, meta):
        self.version = meta['version']

        node = rule['node']
        rule_data = {k: v for k, v in rule.items() if k != "node"}
        action_type = rule['action']

        # 1. 初始化節點的路由表為一個列表
        if node not in self.rules:
            self.rules[node] = []

        # 2. 處理 ADD 動作：直接追加
        if action_type == action.ADD:
            self.rules[node].append(rule_data)
            
        # 3. 處理 REPLACE 動作：按目標網段查找，有則替換，無則追加
        elif action_type == action.REPLACE:
            found = False
            for i, existing_rule in enumerate(self.rules[node]):
                if existing_rule.get('dst_cidr') == rule.get('dst_cidr'):
                    self.rules[node][i] = rule_data
                    found = True
                    break
            if not found:
                self.rules[node].append(rule_data)
                
        # 4. 處理 DEL 動作：精準剔除目標網段匹配的規則
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


    def ExecuteReq(self, client, content_id, time, log_path):
        # 1. 檢查客戶端是否有路由表
        if client not in self.rules or not self.rules[client]:
            LogColor.error(f"[{time}ms] Request Failed: Node {client} 找不到路由規則。")
            return

        # --- 優化點：緩存 Target Node，避免每次都遍歷所有內容 ---
        if not hasattr(self, 'content_locator_cache'):
            self.content_locator_cache = dict()
            
        target_node = None
        if content_id in self.content_locator_cache:
            target_node = self.content_locator_cache[content_id]
        else:
            # 2. 確定誰有這個內容 (尋找 Target Node)
            for node, files in self.content.items():
                if content_id in files:
                    target_node = node
                    self.content_locator_cache[content_id] = target_node
                    break
        
        if not target_node:
            LogColor.error(f"[{time}ms] Request Failed: 內容 '{content_id}' 在網絡中不存在。")
            return

        # 3. 獲取目標節點的 IP
        target_ip = self.ip.get(target_node)
        if not target_ip:
            LogColor.error(f"[{time}ms] Request Failed: 目標節點 {target_node} 沒有 IP 映射。")
            return

        # --- 優化點：緩存 Path，如果拓撲沒變，就不用重新逐跳尋找 ---
        if not hasattr(self, 'path_cache'):
            self.path_cache = dict()
        
        # 如果 self.topology_changed 為 True，代表有節點移動或斷線，清空之前的路徑記憶
        if hasattr(self, 'topology_changed') and self.topology_changed:
            self.path_cache.clear()
            self.topology_changed = False
            
        cache_key = (client, target_node)
        
        if cache_key in self.path_cache:
            # 如果快取裡有路徑，直接拿出來用
            path = self.path_cache[cache_key]
        else:
            # 4. 逐跳转發模擬路徑 (只在拓撲改變，或第一次請求時執行)
            path = [client]
            current_node = client
            max_hops = 20  # 防止環路
            
            while current_node != target_node:
                next_hop = self.find_next_hop(current_node, target_ip)
                
                # 檢查物理圖 G 中是否存在這條邊
                if not next_hop or next_hop not in self.G[current_node]:
                    LogColor.error(f"[{time}ms] 路由失敗於 {current_node}: 無法到達 {target_ip} (下一跳: {next_hop})")
                    return 
                
                # 檢查物理鏈路狀態
                edge_data = self.G[current_node][next_hop]
                if edge_data.get('status', 'UP') != 'UP':
                    LogColor.error(f"[{time}ms] 路由失敗於 {current_node}: 到 {next_hop} 的物理鏈路斷開。")
                    return

                path.append(next_hop)
                current_node = next_hop
                
                if len(path) > max_hops:
                    LogColor.error(f"[{time}ms] 檢測到路由環路: {client} -> {target_node}!")
                    return
            
            # 將成功找到的路徑存入快取
            self.path_cache[cache_key] = path

        # 5. 到達目的地後，計算路徑物理特性
        total_delay = 0.0
        min_bw = float('inf')
        for u, v in zip(path[:-1], path[1:]):
            data = self.G[u][v]
            d = float(data.get('delay_ms', 0))
            b = float(data.get('bw_mbps', 1))
            total_delay += d
            min_bw = min(min_bw, b)

        # 獲取文件大小
        content_info = self.content[target_node][content_id]
        file_size_MB = content_info['filesize']

        # 計算下載時間 (毫秒)
        if min_bw <= 0: min_bw = 0.001
        download_time = (file_size_MB * 8 / min_bw * 1000) + (total_delay * 2)

        # 6. 寫入日誌
        log_row = {
            'time_ms': time,
            'node_id': client,
            'content_id': content_id,
            'file_size_MB': file_size_MB,
            'path': " -> ".join(path),
            'server_node': target_node,
            'latency_ms': total_delay * 2,
            'throughput_mbps': min_bw,
            'http_code': 200,
            'cache_status': 'HIT',
            'download_time': download_time
        }

        self.WriteLog(log_row, log_path)

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
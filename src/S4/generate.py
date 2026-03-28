import random

def generate_sar_traffic(uav_list, main_gs, max_time_ms=60000):
    """
    生成真实的搜救任务数据流：
    - GS 控制信号：全程 10kbps (所有无人机)
    - 低清图像流：全程 10Mbps × 3架 UAV = 30Mbps
    - 高清图像流：
      - UAV_0: 从 180s(3分钟) 开始 40Mbps
      - UAV_1: 从 300s(5分钟) 开始 40Mbps
      - UAV_2: 从 360s(6分钟) 开始 40Mbps
    - 总峰值：30 + 120 = 150Mbps (含控制信号)
    - 每个数据点添加 ±5% 随机扰动
    """
    flows = []  # 数据流列表
    
    # 采样周期：100ms
    sample_interval = 100  # ms
    
    for current_time in range(0, max_time_ms, sample_interval):
        
        # ============ 1. GS 控制信号：每架 UAV 10kbps ============
        control_bw_per_uav = 10.0  # kbps
        perturbation = random.uniform(0.95, 1.05)
        
        for i, uav in enumerate(uav_list):
            flows.append({
                'time': current_time,
                'src': main_gs,
                'dst': uav,
                'flow_type': 'control',
                'bandwidth_mbps': control_bw_per_uav / 1000.0 * perturbation,  # 转换为 Mbps
                'content_id': f'gs_control_{uav}'
            })
        
        # ============ 2. 低清图像流：全程 10Mbps × 3架 ============
        low_res_bw = 10.0  # Mbps per UAV
        
        for i, uav in enumerate(uav_list):
            perturbation = random.uniform(0.95, 1.05)
            flows.append({
                'time': current_time,
                'src': uav,
                'dst': main_gs,
                'flow_type': 'low_res_video',
                'bandwidth_mbps': low_res_bw * perturbation,
                'content_id': f'low_res_img_{uav}'
            })
        
        # ============ 3. 高清图像流：按时间阶段激活 ============
        hd_res_bw = 40.0  # Mbps per UAV
        
        # UAV_0 从 180s (3分钟) 开始发送高清
        if current_time >= 180000:
            perturbation = random.uniform(0.95, 1.05)
            flows.append({
                'time': current_time,
                'src': uav_list[0],
                'dst': main_gs,
                'flow_type': 'hd_video',
                'bandwidth_mbps': hd_res_bw * perturbation,
                'content_id': f'hd_img_{uav_list[0]}'
            })
        
        # UAV_1 从 300s (5分钟) 开始发送高清
        if current_time >= 300000:
            perturbation = random.uniform(0.95, 1.05)
            flows.append({
                'time': current_time,
                'src': uav_list[1],
                'dst': main_gs,
                'flow_type': 'hd_video',
                'bandwidth_mbps': hd_res_bw * perturbation,
                'content_id': f'hd_img_{uav_list[1]}'
            })
        
        # UAV_2 从 360s (6分钟) 开始发送高清
        if current_time >= 360000:
            perturbation = random.uniform(0.95, 1.05)
            flows.append({
                'time': current_time,
                'src': uav_list[2],
                'dst': main_gs,
                'flow_type': 'hd_video',
                'bandwidth_mbps': hd_res_bw * perturbation,
                'content_id': f'hd_img_{uav_list[2]}'
            })
    
    # 按时间排序
    flows.sort(key=lambda x: x['time'])
    return flows
import random

def generate_sar_traffic(uav_list, main_gs, max_time_ms=600000):
    """
    生成 SAR（搜索救援）任务的通信请求。

    模拟三个阶段：
    - 阶段 1（0-30s）：地面站拉取低清图像
    - 阶段 2（30s 后）：地面站拉取 4K 高清视频流
    - 阶段 3（35s 时）：除 UAV_02 外的无人机接收集结命令

    Args:
        uav_list: 无人机列表
        main_gs: 主地面站 ID
        max_time_ms: 最大模拟时间（毫秒）

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
        
        # 阶段 1：常规搜索（0-30s），地面站拉取低清图像
        if current_time < 30000 and current_time % 2000 == 0:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(0, 100),
                    'node_id': main_gs,
                    'content_id': f'low_res_img_{uav}'
                })
        
        # 阶段 2：发现目标（30s 之后），地面站拉取 4K 视频流
        if current_time >= 30000 and current_time % 500 == 0:
            requests.append({
                'time': current_time,
                'node_id': main_gs,
                'content_id': '4k_video_stream'
            })
        
        if current_time == 35000:
            for uav in uav_list:
                # UAV_02 正忙于传输视频，其他无人机前往支援
                if uav != 'UAV_02':
                    requests.append({
                        'time': current_time,
                        'node_id': uav,
                        'content_id': 'c2_converge_cmd'
                    })
        
    # 按 'time' 排序，适配主循环判断
    requests.sort(key=lambda x: x['time'])
    return requests


def generate_uav_requests(uav_list, max_time_ms=600000):
    """
    生成无人机之间的通信请求。

    模拟无人机集群的典型通信模式：
    - 状态共享（每 1500ms）
    - 目标位置信息共享（每 3000ms，20s 后开始）
    - 协作任务请求（每 4500ms，25s 后开始）
    - 紧急情况求助（5% 随机触发）
    - 燃油状态共享（每 5000ms）

    Args:
        uav_list: 无人机列表
        max_time_ms: 最大模拟时间（毫秒）

    Returns:
        按时间排序的请求列表
    """
    requests = []
    
    for current_time in range(0, max_time_ms, 100):
        
        # 无人机之间的状态共享（每1500ms）
        if current_time % 1500 == 0:
            for i, uav in enumerate(uav_list):
                # 随机选择一个其他无人机作为目标
                other_uavs = [v for v in uav_list if v != uav]
                if other_uavs:
                    target_uav = random.choice(other_uavs)
                    requests.append({
                        'time': current_time + random.randint(0, 50),
                        'node_id': uav,
                        'content_id': f'status_update_{target_uav}'
                    })
        
        # 目标位置信息共享（每 3000ms，20s 后开始）
        if current_time % 3000 == 0 and current_time >= 20000:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(0, 100),
                    'node_id': uav,
                    'content_id': 'target_location_update'
                })
        
        # 协作任务请求（每 4500ms，25s 后开始）
        if current_time % 4500 == 0 and current_time >= 25000:
            # 随机选择一个无人机作为任务发起方
            task_initiator = random.choice(uav_list)
            # 随机选择一个无人机作为协作对象
            other_uavs = [u for u in uav_list if u != task_initiator]
            if other_uavs:
                task_partner = random.choice(other_uavs)
                requests.append({
                    'time': current_time,
                    'node_id': task_initiator,
                    'content_id': f'collaboration_request_{task_partner}'
                })
        
        # 紧急情况求助（0.5% 概率随机触发）
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
                other_uavs = [v for v in uav_list if v != uav]
                if other_uavs:
                    target_uav = random.choice(other_uavs)
                    requests.append({
                        'time': current_time + random.randint(0, 50),
                        'node_id': uav,
                        'content_id': f'fuel_status_{target_uav}'
                    })
    
    # 按 'time' 排序，适配主循环判断
    requests.sort(key=lambda x: x['time'])
    return requests
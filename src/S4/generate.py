import random

def generate_sar_traffic(uav_list, main_gs, max_time_ms=600000):
    """
    生成照片侦察任务的通信请求。
    每个 UAV 每 10 秒向地面站发送一张约 12MB 的照片。
    node_id 为 UAV（匹配 S3 UAV→GS 路由方向），content 存储在 GS 上。

    Args:
        uav_list: 无人机列表
        main_gs: 主地面站 ID
        max_time_ms: 最大模拟时间（毫秒）

    Returns:
        按时间排序的请求列表
    """
    requests = []

    # 照片上传：每个 UAV 每 10 秒发起一次照片传输到地面站
    for current_time in range(0, max_time_ms, 10000):
        for i, uav in enumerate(uav_list):
            offset = (i * 200) % 10000  # 错开不同 UAV 避免同时拥塞
            requests.append({
                'time': current_time + offset + random.randint(0, 100),
                'node_id': uav,
                'content_id': f'photo_upload_{uav}'
            })

    # 遥测上传：UAV 每秒向地面站发送遥测数据
    for current_time in range(0, max_time_ms, 100):
        if current_time % 1000 == 0:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(0, 50),
                    'node_id': uav,
                    'content_id': f'telemetry_upload_{uav}'
                })

    requests.sort(key=lambda x: x['time'])
    return requests


def generate_uav_requests(uav_list, max_time_ms=600000):
    """
    生成无人机之间的通信请求。
    模拟无人机集群的协调通信：
    - 状态共享（每 1500ms）
    - 目标位置信息共享（每 3000ms，20s 后开始）
    - 紧急情况求助（0.5% 随机触发）
    - 燃油状态共享（每 5000ms）

    Args:
        uav_list: 无人机列表
        max_time_ms: 最大模拟时间（毫秒）

    Returns:
        按时间排序的请求列表
    """
    requests = []

    for current_time in range(0, max_time_ms, 100):

        # 无人机之间的状态共享（每 1500ms）
        if current_time % 1500 == 0:
            for uav in uav_list:
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

    requests.sort(key=lambda x: x['time'])
    return requests

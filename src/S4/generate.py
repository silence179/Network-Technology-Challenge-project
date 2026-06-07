import random

def generate_sar_traffic(uav_list, main_gs, max_time_ms=600000):
    """
    生成照片侦察任务的通信请求。
    GS 每 10 秒向每个 UAV 请求一张约 12MB 的照片（经 SAT 中转）。
    node_id 为 UAV（匹配 S3 UAV→GS 路由方向），content 存储在 GS 上。

    Args:
        uav_list: 无人机列表
        main_gs: 主地面站 ID
        max_time_ms: 最大模拟时间（毫秒）

    Returns:
        按时间排序的请求列表
    """
    requests = []

    # 照片请求：每个 UAV 每 10 秒向 GS 上传一张照片
    for current_time in range(0, max_time_ms, 10000):
        for i, uav in enumerate(uav_list):
            offset = (i * 200) % 10000  # 错开不同 UAV 避免同时拥塞
            requests.append({
                'time': current_time + offset + random.randint(0, 100),
                'node_id': uav,
                'content_id': f'photo_upload_{uav}'
            })

    # 遥测数据：每个 UAV 每秒向 GS 上传
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

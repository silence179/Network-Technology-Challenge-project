import random
from typing import List, Dict, Optional

# ===================== 常量定义（可配置化）=====================
# 基础遥测请求周期（毫秒）
TELEMETRY_CYCLE_MS = 1000
# 遥测请求时间偏移范围（毫秒）
TELEMETRY_OFFSET_RANGE = (0, 50)
# 默认最大仿真时间（毫秒）
DEFAULT_MAX_TIME_MS = 60000
# 阶段1：常规搜索（低清图像）配置
LOW_RES_STAGE_END_MS = 30000
LOW_RES_CYCLE_MS = 2000
LOW_RES_OFFSET_RANGE = (0, 100)
# 阶段2：目标跟踪（4K视频）配置
HIGH_RES_STAGE_START_MS = 30000
HIGH_RES_CYCLE_MS = 500
# 汇聚指令触发时间（毫秒）
CONVERGE_CMD_TRIGGER_MS = 35000
CONVERGE_EXCLUDE_UAV = "UAV_02"
CONVERGE_CONTENT_ID = "c2_converge_cmd"


def generate_sar_traffic(
    uav_list: List[str],
    main_gs: str,
    max_time_ms: int = DEFAULT_MAX_TIME_MS,
    random_seed: Optional[int] = 42,
    enable_extended_stages: bool = False
) -> List[Dict]:
    """
    生成SAR场景下无人机与地面站的业务流量请求
    
    Args:
        uav_list: 无人机ID列表
        main_gs: 主地面站ID
        max_time_ms: 最大仿真时间（毫秒），默认60000ms（60秒）
        random_seed: 随机种子（保证结果可复现），默认42
        enable_extended_stages: 是否启用扩展阶段（低清图像+4K视频+汇聚指令）
    
    Returns:
        按时间排序的流量请求列表
    
    Raises:
        ValueError: 输入参数不合法时抛出
    """
    # 1. 参数合法性校验
    if not isinstance(uav_list, list) or not uav_list:
        raise ValueError("uav_list必须是非空列表")
    if not isinstance(main_gs, str) or not main_gs.strip():
        raise ValueError("main_gs必须是非空字符串")
    if not isinstance(max_time_ms, int) or max_time_ms <= 0:
        raise ValueError("max_time_ms必须是正整数")
    
    # 2. 设置随机种子，保证结果可复现
    random.seed(random_seed)
    
    # 3. 初始化请求列表
    requests = []
    
    # 4. 生成流量请求（优化循环步长，减少无效迭代）
    # 核心逻辑：按最小周期步长迭代，避免100ms空跑
    min_cycle = TELEMETRY_CYCLE_MS
    if enable_extended_stages:
        min_cycle = min(TELEMETRY_CYCLE_MS, LOW_RES_CYCLE_MS, HIGH_RES_CYCLE_MS)
    
    for current_time in range(0, max_time_ms, min_cycle):
        # 基础遥测请求（核心逻辑，始终启用）
        if current_time % TELEMETRY_CYCLE_MS == 0:
            for uav in uav_list:
                requests.append({
                    'time': current_time + random.randint(*TELEMETRY_OFFSET_RANGE),
                    'node_id': main_gs,
                    'content_id': f'telemetry_{uav}'
                })
        
        # 扩展阶段：仅当启用时执行
        if enable_extended_stages:
            # 阶段1：常规搜索 - 低清图像（0~30秒）
            if current_time < LOW_RES_STAGE_END_MS and current_time % LOW_RES_CYCLE_MS == 0:
                for uav in uav_list:
                    requests.append({
                        'time': current_time + random.randint(*LOW_RES_OFFSET_RANGE),
                        'node_id': main_gs,
                        'content_id': f'low_res_img_{uav}'
                    })
            
            # 阶段2：目标跟踪 - 4K视频（30秒后）
            if current_time >= HIGH_RES_STAGE_START_MS and current_time % HIGH_RES_CYCLE_MS == 0:
                requests.append({
                    'time': current_time,
                    'node_id': main_gs,
                    'content_id': '4k_video_stream'
                })
            
            # 阶段3：汇聚指令（35秒时触发）
            if current_time == CONVERGE_CMD_TRIGGER_MS:
                for uav in uav_list:
                    if uav != CONVERGE_EXCLUDE_UAV:
                        requests.append({
                            'time': current_time,
                            'node_id': uav,
                            'content_id': CONVERGE_CONTENT_ID
                        })
    
    # 5. 按时间排序，保证请求时序正确
    requests.sort(key=lambda x: x['time'])
    
    return requests


# ===================== 测试示例（可选）=====================
if __name__ == "__main__":
    # 测试基础功能
    test_uavs = ["UAV_01", "UAV_02", "UAV_03"]
    test_gs = "GS_MAIN"
    
    # 测试基础遥测流量
    basic_requests = generate_sar_traffic(test_uavs, test_gs, max_time_ms=5000)
    print(f"基础遥测请求数：{len(basic_requests)}")
    print("前3条请求示例：")
    for req in basic_requests[:3]:
        print(req)
    
    # 测试扩展阶段流量
    extended_requests = generate_sar_traffic(
        test_uavs, test_gs, max_time_ms=40000, enable_extended_stages=True
    )
    print(f"\n扩展阶段请求数：{len(extended_requests)}")
    print("35秒附近的汇聚指令请求：")
    converge_cmds = [r for r in extended_requests if r['content_id'] == CONVERGE_CONTENT_ID]
    for cmd in converge_cmds:
        print(cmd)
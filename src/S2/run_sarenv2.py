import os
import csv
import math
from datetime import datetime
import numpy as np
from pymap3d import enu2ecef

import sarenv
from sarenv.analytics.paths import generate_spiral_path
from sarenv import LostPersonLocationGenerator

# 1. 配置参数
dataset_dir = "sarenv_dataset"
num_uavs = 3
search_radius_m = 500
altitude_m = 50
fov_deg = 60
overlap = 0.2
path_point_spacing_m = 10
uav_speed_mps = 15
detection_range_m = 30  # 相机检测范围（米）

ANCHOR_LAT = 30.0
ANCHOR_LON = 104.0
ANCHOR_ALT = 500.0

GS_ECEF_X, GS_ECEF_Y, GS_ECEF_Z = enu2ecef(0, 0, 0, ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT, deg=True)

CHUNK_DURATION_MS = 60000  # 60秒
TIME_STEP_MS = 100  # 10Hz
TOTAL_DURATION_MS = 420000  # 总时长7分钟

# ==================== 新增：生成失联人员位置 ====================
print("[S2] 加载环境并生成失联人员位置...")
loader = sarenv.DatasetLoader(dataset_dir)
env_item = loader.load_environment("large")

# 生成20个失联人员位置（基于真实行为模型）
# 修改：参数名从 num_locations 改为 n，random_ratio 改为 percent_random_samples
victim_generator = LostPersonLocationGenerator(env_item)
victims = victim_generator.generate_locations(n=20, percent_random_samples=0)

# 转换为ENU坐标（相对于anchor）
victims_enu = [(v.x, v.y) for v in victims]
print(f"[S2] 已生成 {len(victims_enu)} 个失联人员位置")

# 2. 生成无人机路径
spiral_paths = generate_spiral_path(
    center_x=0, center_y=0,
    max_radius=search_radius_m,
    fov_deg=fov_deg,
    altitude=altitude_m,
    overlap=overlap,
    num_drones=num_uavs,
    path_point_spacing_m=path_point_spacing_m,
)

# -------------------------
# 3. 预计算轨迹（带时间戳 + 检测逻辑）
# -------------------------
def interpolate_path_to_10hz(path, speed_mps, victims_enu, detection_range):
    """
    插值路径到10Hz，并加入失联人员检测逻辑
    返回: {time_ms: (x, y, heading, role, detected_victim_ids)}
    """
    coords = list(path.coords)
    trajectory = {}
    time_ms = 0
    detected_victims = set()  # 已发现的受害者ID
    cache_until_ms = -1  # CACHE模式持续到什么时候
    
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dist = math.hypot(x2 - x1, y2 - y1)
        duration_ms = int((dist / speed_mps) * 1000)
        steps = max(1, duration_ms // TIME_STEP_MS)
        
        dx = x2 - x1
        dy = y2 - y1
        heading = math.degrees(math.atan2(dx, dy)) % 360
        
        for step in range(steps):
            t = time_ms + step * TIME_STEP_MS
            ratio = step / steps
            x = x1 + (x2 - x1) * ratio
            y = y1 + (y2 - y1) * ratio
            
            # ==================== 检测逻辑 ====================
            newly_detected = []
            
            # 检查是否发现新的受害者
            for vid, (vx, vy) in enumerate(victims_enu):
                if vid in detected_victims:
                    continue
                d = math.hypot(x - vx, y - vy)
                if d < detection_range:
                    detected_victims.add(vid)
                    newly_detected.append(vid)
                    # 发现后进入CACHE模式10秒（悬停传输）
                    cache_until_ms = max(cache_until_ms, t + 10000)
                    print(f"  [t={t}ms] 发现受害者 #{vid}！位置({vx:.1f}, {vy:.1f})，距离{d:.1f}m")
            
            # 确定当前角色
            if t < cache_until_ms:
                role = "CACHE"  # 悬停传输模式
            else:
                role = "RELAY"  # 正常搜索模式
            
            trajectory[t] = (x, y, heading, role, list(detected_victims))
        
        time_ms += duration_ms
    
    # 记录最后状态
    last_pos = coords[-1] if coords else (0, 0)
    trajectory['last_time'] = time_ms
    trajectory['last_pos'] = (last_pos[0], last_pos[1], heading)
    trajectory['final_detected'] = list(detected_victims)
    
    return trajectory

# 为每架无人机生成轨迹
print("[S2] 生成无人机轨迹（带检测）...")
uav_trajectories = [
    interpolate_path_to_10hz(p, uav_speed_mps, victims_enu, detection_range_m) 
    for p in spiral_paths
]

# 打印统计
for i, traj in enumerate(uav_trajectories):
    final_detected = traj.get('final_detected', [])
    print(f"  UAV_{i+1}: 共发现 {len(final_detected)} 个目标")

# 4. 按固定时间循环生成数据
uav_ips = [f"10.0.0.{2+i}" for i in range(num_uavs)]
gs_ip = "10.0.0.1"

INIT_BATTERY = 100.0
BATTERY_DRAIN_PER_SEC = 0.1

def get_uav_state_at_time(traj, time_ms):
    """获取指定时间的UAV状态"""
    if time_ms in traj:
        x, y, heading, role, _ = traj[time_ms]
        return x, y, heading, role
    elif time_ms > traj['last_time']:
        # 已到达终点，保持最后位置悬停（CACHE模式）
        x, y, heading = traj['last_pos']
        return x, y, heading, "CACHE"
    else:
        # 时间点在步进之间，找前一个已知点
        known_times = [t for t in traj.keys() if isinstance(t, int) and t <= time_ms]
        if known_times:
            closest_time = max(known_times)
            x, y, heading, role, _ = traj[closest_time]
            return x, y, heading, role
        else:
            x, y, heading = traj['last_pos']
            return x, y, heading, "RELAY"

def write_chunk(data, start_ms):
    """写入切片文件"""
    end_ms = start_ms + CHUNK_DURATION_MS
    filename = f"uav_trace_{start_ms}_{end_ms}.csv"
    fieldnames = ["time_ms", "node_id", "role", "type", "ecef_x", "ecef_y", "ecef_z", 
                  "ip", "heading_deg", "battery_pct"]
    
    # 确保目录存在
    os.makedirs("traces", exist_ok=True)
    filepath = os.path.join("traces", filename)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"[SUCCESS] {filepath} ({len(data)}行)")

# 按时间切片生成
current_chunk = []
chunk_start_ms = 0

for time_ms in range(0, TOTAL_DURATION_MS, TIME_STEP_MS):
    # 检查是否需要切分
    if time_ms >= chunk_start_ms + CHUNK_DURATION_MS:
        write_chunk(current_chunk, chunk_start_ms)
        current_chunk = []
        chunk_start_ms = time_ms
    
    # GS数据（每帧都生成）
    current_chunk.append({
        "time_ms": time_ms,
        "node_id": "GS_01",
        "role": "CLIENT",
        "type": "GS",
        "ecef_x": round(GS_ECEF_X, 1),
        "ecef_y": round(GS_ECEF_Y, 1),
        "ecef_z": round(GS_ECEF_Z, 1),
        "ip": gs_ip,
        "heading_deg": -1.0,
        "battery_pct": -1
    })
    
    # UAV数据
    for agent_id in range(num_uavs):
        x, y, heading, role = get_uav_state_at_time(uav_trajectories[agent_id], time_ms)
        
        ecef_x, ecef_y, ecef_z = enu2ecef(x, y, altitude_m, ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT, deg=True)
        
        elapsed_sec = time_ms / 1000
        battery_pct = max(0, int(INIT_BATTERY - elapsed_sec * BATTERY_DRAIN_PER_SEC))
        
        current_chunk.append({
            "time_ms": time_ms,
            "node_id": f"UAV_{agent_id+1:02d}",
            "role": role,
            "type": "UAV",
            "ecef_x": round(ecef_x, 1),
            "ecef_y": round(ecef_y, 1),
            "ecef_z": round(ecef_z, 1),
            "ip": uav_ips[agent_id],
            "heading_deg": round(heading, 1),
            "battery_pct": battery_pct
        })

# 写入最后一个切片
if current_chunk:
    write_chunk(current_chunk, chunk_start_ms)

print(f"[DONE] 仿真时长：{TOTAL_DURATION_MS}ms")

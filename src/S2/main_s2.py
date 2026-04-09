import os
import csv
import math
import random
from pymap3d import enu2ecef  # 用于将站心坐标系(ENU)转换为地心轴坐标系(ECEF)

import sarenv
from sarenv.analytics.paths import generate_spiral_path  # 专门用于搜救场景的螺旋路径生成函数

# -------------------------
# 1. 地理与仿真配置 (Geographic & Simulation Config)
# -------------------------
ANCHOR_LAT = 30  
ANCHOR_LON = 104 
ANCHOR_ALT = 459

NUM_UAVS = 5               # 无人机数量
SEARCH_RADIUS_M = 2500     # 搜索覆盖半径 2.5km
ALTITUDE_M = 50            # 飞行相对高度 50m
DETECTION_RANGE_M = 60     # 无人机传感器检测半径
UAV_SPEED_MPS = 15         # 15m/s 恒定巡航速度

TIME_STEP_MS = 100         # 100ms 采样周期 (10Hz)
TOTAL_DURATION_MS = 600000 # 10 分钟模拟时长 (10 * 60 * 1000)

# -------------------------
# 2. 受害者生成逻辑 (Victim Generation)
# -------------------------
def generate_victims_in_sichuan(num, radius):
    victims = []
    for i in range(num):
        r = radius * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        vx = r * math.cos(theta)
        vy = r * math.sin(theta)
        victims.append((vx, vy))
    return victims

print(f"[S1] 坐标锚点已设为：({ANCHOR_LAT}, {ANCHOR_LON}, {ANCHOR_ALT}m)")
victims_enu = generate_victims_in_sichuan(50, SEARCH_RADIUS_M)
print(f"[S1] 已布设 {len(victims_enu)} 个受害者。")

# -------------------------
# 3. 路径规划 (Path Planning)
# -------------------------
print("[S2] 规划协同螺旋路径...")
spiral_paths = generate_spiral_path(
    center_x=0, center_y=0,
    max_radius=SEARCH_RADIUS_M,
    fov_deg=60, altitude=ALTITUDE_M,
    overlap=0.3, num_drones=NUM_UAVS,
    path_point_spacing_m=30, 
)

# -------------------------
# 4. 平滑插值与检测逻辑 (Interpolation & Detection)
# -------------------------
def process_uav_mission(path, victims_list):
    coords = list(path.coords)
    traj_data = {}        
    current_time_ms = 0.0
    detected_ids = set()  
    cache_until_ms = -1.0 
    
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        seg_ms = (dist / UAV_SPEED_MPS) * 1000  
        
        start_ms = current_time_ms
        end_ms = current_time_ms + seg_ms
        t_sample = math.ceil(start_ms / TIME_STEP_MS) * TIME_STEP_MS
        
        while t_sample <= end_ms and t_sample <= TOTAL_DURATION_MS:
            ratio = (t_sample - start_ms) / seg_ms if seg_ms > 0 else 0
            curr_x = p1[0] + (p2[0] - p1[0]) * ratio
            curr_y = p1[1] + (p2[1] - p1[1]) * ratio
            angle = math.degrees(math.atan2(p2[0]-p1[0], p2[1]-p1[1])) % 360
            
            for vid, (vx, vy) in enumerate(victims_list):
                if vid not in detected_ids:
                    if math.hypot(curr_x - vx, curr_y - vy) < DETECTION_RANGE_M:
                        detected_ids.add(vid)
                        cache_until_ms = max(cache_until_ms, t_sample + 10000)
                        print(f"  [t={int(t_sample)}ms] UAV 发现目标 #{vid}！")
            
            role = "CACHE" if t_sample < cache_until_ms else "RELAY"
            traj_data[int(t_sample)] = (curr_x, curr_y, angle, role)
            t_sample += TIME_STEP_MS
            
        current_time_ms = end_ms
        if current_time_ms > TOTAL_DURATION_MS: break
    
    # 悬停补全
    last_pos = (coords[-1][0], coords[-1][1], 0, "RELAY")
    for t in range(0, TOTAL_DURATION_MS + TIME_STEP_MS, TIME_STEP_MS):
        if t not in traj_data:
            traj_data[t] = traj_data.get(t - TIME_STEP_MS, last_pos)
            
    return traj_data, len(detected_ids)

print("[S3] 计算平滑飞行轨迹...")
uav_results = [process_uav_mission(p, victims_enu) for p in spiral_paths]

# -------------------------
# 5. 生成切片 CSV 文件 (Export Data)
# -------------------------

# 【修改点】获取当前脚本的绝对路径，并向上退两级定位到根目录，然后再进入 S3/uav_trace
current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(current_dir, "..", "S3", "uav_trace"))

# 确保目标文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

GS_ECEF = enu2ecef(0, 0, 0, ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT, deg=True)

# 定义切片时长为 60 秒 (60,000 ms)
CHUNK_DURATION_MS = 60000  

print(f"[S4] 正在导出切片 CSV 至: {OUTPUT_DIR}")

fieldnames = ["time_ms", "node_id", "role", "type", "ecef_x", "ecef_y", "ecef_z", "ip", "heading_deg", "battery_pct"]

# 按照 60 秒进行循环切片
for chunk_start in range(0, TOTAL_DURATION_MS, CHUNK_DURATION_MS):
    # 计算当前切片的结束时间，例如 0 ~ 59999
    chunk_end = min(chunk_start + CHUNK_DURATION_MS - 1, TOTAL_DURATION_MS - 1)
    
    filename = f"uav_trace_{chunk_start}_{chunk_end}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 遍历当前 60 秒切片内的时间戳
        for t in range(chunk_start, chunk_start + CHUNK_DURATION_MS, TIME_STEP_MS):
            if t >= TOTAL_DURATION_MS:
                break
                
            # 1. 地面站
            writer.writerow({
                "time_ms": t, "node_id": "GS_01", "role": "CLIENT", "type": "GS",
                "ecef_x": round(GS_ECEF[0], 1), "ecef_y": round(GS_ECEF[1], 1), "ecef_z": round(GS_ECEF[2], 1),
                "ip": "10.0.0.1", "heading_deg": -1.0, "battery_pct": -1
            })
            
            # 2. 三架无人机
            for i in range(NUM_UAVS):
                traj, _ = uav_results[i]
                
                # 获取位置信息
                ux, uy, uh, urole = traj[t]
                ex, ey, ez = enu2ecef(ux, uy, ALTITUDE_M, ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT, deg=True)
                batt = round(max(0.0, 100 - (t / 1000 * 0.1)), 1)
                
                writer.writerow({
                    "time_ms": t, "node_id": f"UAV_{i+1:02d}", "role": urole, "type": "UAV",
                    "ecef_x": round(ex, 1), "ecef_y": round(ey, 1), "ecef_z": round(ez, 1),
                    "ip": f"10.0.0.{2+i}", "heading_deg": round(uh, 1), "battery_pct": batt
                })

print("\n" + "="*30)
for i, (_, count) in enumerate(uav_results):
    print(f"UAV_{i+1} 搜救总结: 成功发现 {count} 名失联人员")
print("="*30)
print(f"[DONE] 仿真结束。切片文件已存入: {OUTPUT_DIR}")

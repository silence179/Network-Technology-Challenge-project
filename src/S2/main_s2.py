import os
import sys
import csv
import math
import random
import json
import subprocess
from pymap3d import enu2ecef  # 用于将站心坐标系(ENU)转换为地心轴坐标系(ECEF)

# ======================== SAREnv 自动安装 ========================
SARENV_REPO_URL = "https://github.com/namurproject/SAREnv.git" 

S2_DIR = os.path.dirname(os.path.abspath(__file__))
SARENV_DIR = os.path.join(S2_DIR, "SAREnv")
sys.path.insert(0, SARENV_DIR)

def _ensure_sarenv():
    """自动检测并安装 sarenv 包"""
    # 先尝试直接导入
    sarenv_path = os.path.join(SARENV_DIR, "sarenv")
    if os.path.isdir(sarenv_path):
        return  # 已存在

    print("[S2] sarenv 未安装，正在自动下载...")
    print(f"     仓库: {SARENV_REPO_URL}")

    # 若 SAREnv 目录非空则先清理
    if os.path.isdir(SARENV_DIR) and os.listdir(SARENV_DIR):
        import shutil
        for item in os.listdir(SARENV_DIR):
            item_path = os.path.join(SARENV_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)

    # Clone 仓库到临时目录，再移动到 SAREnv/
    tmp_dir = os.path.join(S2_DIR, "_sarenv_tmp")
    try:
        subprocess.run(
            ["git", "clone", SARENV_REPO_URL, tmp_dir],
            check=True, capture_output=True, text=True
        )
        # 安装依赖
        req_file = os.path.join(tmp_dir, "requirements.txt")
        if os.path.exists(req_file):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_file],
                check=True
            )
        # pip install -e .
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", tmp_dir],
            check=True
        )
        print("[S2] sarenv 安装成功！")
    except subprocess.CalledProcessError as e:
        print(f"[S2] 自动安装失败: {e}")
        print(f"[S2] 请手动执行:")
        print(f"      git clone {SARENV_REPO_URL}")
        print(f"      cd sarenv && pip install -r requirements.txt && pip install -e .")
        sys.exit(1)
    finally:
        if os.path.isdir(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

_ensure_sarenv()

import sarenv
from sarenv.analytics.paths import generate_spiral_path  # 专门用于搜救场景的螺旋路径生成函数

# -------------------------
# 1. 地理与仿真配置 (Geographic & Simulation Config)
# -------------------------
ANCHOR_LAT = 30  
ANCHOR_LON = 104 
ANCHOR_ALT = 459

NUM_UAVS = 50               # 无人机数量
SEARCH_RADIUS_M = 2500     # 搜索覆盖半径 2.5km
ALTITUDE_M = 50            # 飞行相对高度 50m
DETECTION_RANGE_M = 60     # 无人机传感器检测半径
UAV_SPEED_MPS = 15         # 15m/s 恒定巡航速度

TIME_STEP_MS = 100         # 100ms 采样周期 (10Hz)
TOTAL_DURATION_MS = 600000 # 10 分钟模拟时长 (10 * 60 * 1000)

# 从环境变量加载用户配置（前端传入，覆盖默认值）
_S2_CONFIG_FILE = os.environ.get('S2_CONFIG_FILE', '')
if _S2_CONFIG_FILE and os.path.exists(_S2_CONFIG_FILE):
    with open(_S2_CONFIG_FILE, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    if 'anchor_lat' in _cfg:     ANCHOR_LAT = float(_cfg['anchor_lat'])
    if 'anchor_lon' in _cfg:     ANCHOR_LON = float(_cfg['anchor_lon'])
    if 'anchor_alt' in _cfg:     ANCHOR_ALT = float(_cfg['anchor_alt'])
    if 'num_uavs' in _cfg:       NUM_UAVS = int(_cfg['num_uavs'])
    if 'search_radius' in _cfg:  SEARCH_RADIUS_M = float(_cfg['search_radius'])
    if 'altitude' in _cfg:       ALTITUDE_M = float(_cfg['altitude'])
    if 'detection_range' in _cfg: DETECTION_RANGE_M = float(_cfg['detection_range'])
    if 'uav_speed' in _cfg:      UAV_SPEED_MPS = float(_cfg['uav_speed'])
    if 'duration_ms' in _cfg:    TOTAL_DURATION_MS = int(_cfg['duration_ms'])
    print(f"✅ 已加载用户配置: UAV={NUM_UAVS}, DUR={TOTAL_DURATION_MS}ms, OBS=({ANCHOR_LAT},{ANCHOR_LON},{ANCHOR_ALT})")

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
# 5. 导出切片 CSV (Export Data)
# -------------------------

# 获取当前脚本的绝对路径，向上退一级到 src，再进入对应目录
current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.abspath(os.path.join(current_dir, "..", "S3", "traces"))
VIS_UAV_TRACE_DIR = os.path.abspath(os.path.join(current_dir, "..", "vis", "public", "data", "uav_trace"))

CHUNK_DURATION_MS = 60000  # 每个切片 60 秒
GS_ECEF = enu2ecef(0, 0, 0, ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT, deg=True)
fieldnames = ["time_ms", "node_id", "role", "type", "ecef_x", "ecef_y", "ecef_z", "ip", "heading_deg", "battery_pct"]


def split_and_save_uav_csv(uav_results, output_dir):
    """按60秒切片保存 UAV CSV"""
    os.makedirs(output_dir, exist_ok=True)
    total_chunks = TOTAL_DURATION_MS // CHUNK_DURATION_MS

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * CHUNK_DURATION_MS
        chunk_end = min(chunk_start + CHUNK_DURATION_MS - 1, TOTAL_DURATION_MS - 1)

        filename = f"uav_trace_{chunk_start}_{chunk_end}_{NUM_UAVS}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for t in range(chunk_start, chunk_start + CHUNK_DURATION_MS, TIME_STEP_MS):
                if t >= TOTAL_DURATION_MS:
                    break

                # 地面站
                writer.writerow({
                    "time_ms": t, "node_id": "GS_01", "role": "CLIENT", "type": "GS",
                    "ecef_x": round(GS_ECEF[0], 1), "ecef_y": round(GS_ECEF[1], 1), "ecef_z": round(GS_ECEF[2], 1),
                    "ip": "10.0.0.1", "heading_deg": -1.0, "battery_pct": -1
                })

                # 无人机
                for i in range(NUM_UAVS):
                    traj, _ = uav_results[i]
                    ux, uy, uh, urole = traj[t]
                    ex, ey, ez = enu2ecef(ux, uy, ALTITUDE_M, ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT, deg=True)
                    batt = round(max(0.0, 100 - (t / 1000 * 0.1)), 1)

                    writer.writerow({
                        "time_ms": t, "node_id": f"UAV_{i+1:02d}", "role": urole, "type": "UAV",
                        "ecef_x": round(ex, 1), "ecef_y": round(ey, 1), "ecef_z": round(ez, 1),
                        "ip": f"10.0.0.{2+i}", "heading_deg": round(uh, 1), "battery_pct": batt
                    })

        print(f"💾 保存切片文件：{filename}")


# 输出到 S3 模块
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"uav_trace_{NUM_UAVS}")
print(f"[S4] 正在导出切片 CSV 至 S3: {OUTPUT_DIR}")
split_and_save_uav_csv(uav_results, OUTPUT_DIR)

# 同时输出到 vis 前端
vis_dir = VIS_UAV_TRACE_DIR
os.makedirs(vis_dir, exist_ok=True)
print(f"[S4] 正在导出切片 CSV 至 vis: {vis_dir}")
split_and_save_uav_csv(uav_results, vis_dir)

print("\n" + "="*30)
for i, (_, count) in enumerate(uav_results):
    print(f"UAV_{i+1} 搜救总结: 成功发现 {count} 名失联人员")
print("="*30)
print(f"[DONE] 仿真结束。切片文件已存入: {OUTPUT_DIR}")

from skyfield.api import load, EarthSatellite, Topos, wgs84
from skyfield.framelib import itrs
from skyfield.nutationlib import iau2000b
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import os

# ======================== 全局配置（需与S2协商确认）========================
# 1. 时间配置（T0时刻：仿真起始时间，与S2保持一致）
T0_UTC = datetime(2026, 3, 5, 8, 0, 0)  # 示例：2026-03-05 8:00:00 UTC
SIM_DURATION_SEC = 600  # 仿真总时长（10分钟）
TIME_STEP_SEC = 1  # 时间步长（1秒/帧）
MS_PER_SEC = 1000  # 毫秒转换系数

# 2. 救援区域配置（观察点：与S2选定的救援中心一致）
OBS_LAT = 30.0  # 救援中心纬度（示例：四川某地）
OBS_LON = 104.0  # 救援中心经度
OBS_ELE = 459.0  # 救援中心海拔（米）

# 3. 卫星筛选配置
MIN_ALT_DEG = 0  # 最小仰角（地平线以上）
MAX_DIST_KM = 2000  # 最大距离（2000km）
MAX_SAT_COUNT = 10  # 最终输出卫星数量
IP_PREFIX = "10.0.3."  # 卫星IP前缀

# 4. 文件配置
# 获取当前代码文件的绝对路径（不受运行目录影响）
CODE_FILE_PATH = os.path.abspath(__file__)
# 获取代码文件所在的目录（路径基准）
CODE_DIR = os.path.dirname(CODE_FILE_PATH)
CELESTRAK_STARLINK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" # 星链TLE数据源
TLE_FILE = os.path.join(CODE_DIR, "starlink.tle") # 本地TLE文件路径
OUTPUT_DIR = os.path.join(
    CODE_DIR,
    "StarCDN_Project",
    "data",
    "traces"
)  # 输出目录
CHUNK_DURATION_SEC = 60  # 每个文件的时间切片（60秒）

# 5. 动态筛选配置
DYNAMIC_FILTER_INTERVAL_SEC = 60  # 动态筛选时间窗口（每60秒重新筛选一次）
RESELECT_SAT_COUNT = 10  # 每次动态筛选保留的卫星数

# ======================== 工具函数 ========================
'''
def download_latest_tle():
    """
    从CelesTrak下载最新的Starlink TLE数据，并保存到本地starlink.tle文件
    """
    try:
        # 发送请求获取TLE数据（添加超时和用户代理，避免请求被拦截）
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(CELESTRAK_STARLINK_TLE_URL, headers=headers, timeout=30)
        # 检查请求是否成功
        response.raise_for_status()
        
        # 将获取的TLE数据写入本地文件（覆盖原有内容）
        with open(TLE_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print(f"✅ 成功下载最新Starlink TLE数据，已更新 {TLE_FILE}")
        return True
    
    except requests.exceptions.RequestException as e:
        # 网络请求失败时的异常处理
        print(f"❌ 下载TLE数据失败：{e}")
        # 如果本地已有旧的TLE文件，提示并使用旧文件
        if os.path.exists(TLE_FILE):
            print(f"⚠️ 将使用本地已有的 {TLE_FILE} 文件继续运行")
            return True
        else:
            print(f"❌ 本地无TLE文件且下载失败，程序无法继续")
            return False
'''

def init_time_scale():
    """初始化Skyfield时间标尺并返回T0时刻对象"""
    ts = load.timescale()
    t0 = ts.utc(
        T0_UTC.year, T0_UTC.month, T0_UTC.day,
        T0_UTC.hour, T0_UTC.minute, T0_UTC.second
    )
    return ts, t0

def filter_visible_satellites(all_starlink_sats, observer, current_t):
    """
    单时间点筛选可见卫星
    筛选逻辑：当前时刻仰角>MIN_ALT_DEG 或 距离<MAX_DIST_KM
    """
    visible_sats = []
    for sat in all_starlink_sats:
        diff = sat - observer
        topo = diff.at(current_t)
        alt_deg = topo.altaz()[0].degrees
        dist_km = topo.distance().km

        if alt_deg > MIN_ALT_DEG or dist_km < MAX_DIST_KM:
            visible_sats.append((dist_km, sat))

    # 按距离排序取前N颗
    visible_sats_sorted = sorted(visible_sats, key=lambda x: x[0])
    selected_sats = visible_sats_sorted[:RESELECT_SAT_COUNT]
    
    # 生成动态元数据（保证node_id/IP相对稳定，优先复用已有ID）
    sat_metadata = []
    for idx, (dist_km, sat) in enumerate(selected_sats, 1):
        # 提取卫星唯一标识（NORAD编号），避免重复命名
        norad_id = sat.model.satnum
        sat_id = f"SAT_{norad_id:05d}"  # 用NORAD编号替代顺序号，保证唯一性
        ip = f"{IP_PREFIX}{norad_id % 255}"  # 基于NORAD编号生成IP，避免冲突
        
        sat_metadata.append({
            "node_id": sat_id,
            "name": sat.name.strip(),
            "ip": ip,
            "orbit_id": -1,
            "satellite_obj": sat,
            "norad_id": norad_id,  # 新增NORAD编号，便于追踪
            "current_dist_km": round(dist_km, 2)
        })
    return sat_metadata

def load_and_filter_satellites(t0, observer): #旧的筛选逻辑，静态筛选
    """
    加载TLE数据并筛选符合条件的卫星
    筛选逻辑：T0时刻仰角>0° 或 距离<2000km，按距离排序取前MAX_SAT_COUNT颗
    """
    # 加载所有卫星
    satellites = load.tle_file(TLE_FILE)
    starlink_sats = [sat for sat in satellites if "STARLINK" in sat.name.upper()]
    print(f"📡 加载到 {len(starlink_sats)} 颗Starlink卫星")

    # 筛选可见卫星
    visible_sats = []
    for sat in starlink_sats:
        diff = sat - observer
        topo = diff.at(t0)
        alt_deg = topo.altaz()[0].degrees
        dist_km = topo.distance().km

        # 满足任一条件即保留
        if alt_deg > MIN_ALT_DEG or dist_km < MAX_DIST_KM:
            visible_sats.append((dist_km, sat))

    # 按距离排序，取前N颗
    visible_sats_sorted = sorted(visible_sats, key=lambda x: x[0])
    selected_sats = visible_sats_sorted[:MAX_SAT_COUNT]
    print(f"✅ 筛选出 {len(selected_sats)} 颗符合条件的卫星（按距离排序）")

    # 生成卫星元数据（ID、IP等）
    sat_metadata = []
    for idx, (dist_km, sat) in enumerate(selected_sats, 1):
        sat_id = f"SAT_{idx:02d}"
        ip = f"{IP_PREFIX}{idx}"
        # orbit_id暂填-1（可后续优化）
        sat_metadata.append({
            "node_id": sat_id,
            "name": sat.name.strip(),
            "ip": ip,
            "orbit_id": -1,
            "satellite_obj": sat
        })
    return sat_metadata

def calculate_dynamic_sat_trajectory(all_starlink_sats, ts, t0, observer):
    """
    动态计算卫星轨迹：每DYNAMIC_FILTER_INTERVAL_SEC秒重新筛选可见卫星
    """
    all_traces = []
    total_steps = SIM_DURATION_SEC // TIME_STEP_SEC

    # 预加载所有Starlink卫星（避免重复加载TLE）
    print(f"📡 预加载 {len(all_starlink_sats)} 颗Starlink卫星，开始动态轨迹计算...")

    for step in range(total_steps):
        current_sec = step * TIME_STEP_SEC
        current_time_ms = current_sec * MS_PER_SEC
        current_t = t0 + timedelta(seconds=current_sec)

        # 每N秒重新筛选一次可见卫星
        if current_sec % DYNAMIC_FILTER_INTERVAL_SEC == 0:
            current_sat_metadata = filter_visible_satellites(all_starlink_sats, observer, current_t)
            print(f"⏱️  时间 {current_sec}秒：筛选出 {len(current_sat_metadata)} 颗可见卫星")

        # 计算当前可见卫星的轨迹
        for sat_info in current_sat_metadata:
            sat = sat_info["satellite_obj"]
            geocentric = sat.at(current_t)
            ecef_xyz_m = geocentric.frame_xyz(itrs).m
            ecef_x, ecef_y, ecef_z = ecef_xyz_m

            subpoint = wgs84.subpoint(geocentric)
            altitude_km = subpoint.elevation.km

            trace = {
                "time_ms": current_time_ms,
                "node_id": sat_info["node_id"],
                "name": sat_info["name"],
                "type": "SAT",
                "ecef_x": round(ecef_x, 2),
                "ecef_y": round(ecef_y, 2),
                "ecef_z": round(ecef_z, 2),
                "altitude_km": round(altitude_km, 2),
                "orbit_id": sat_info["orbit_id"],
                "ip": sat_info["ip"],
                "norad_id": sat_info["norad_id"],  # 新增字段：卫星唯一标识
                "distance_km": sat_info["current_dist_km"]  # 新增字段：当前距离
            }
            all_traces.append(trace)

        # 进度提示（每小时输出一次）
        if current_sec % 3600 == 0 and current_sec > 0:
            print(f"🚀 已完成 {current_sec/3600} 小时轨迹计算，累计 {len(all_traces)} 条记录")

    print(f"📊 完成 {total_steps} 个时间步的轨迹计算，共 {len(all_traces)} 条记录")
    return pd.DataFrame(all_traces)

def calculate_sat_trajectory(sat_metadata, ts, t0): #旧的轨迹计算
    """
    计算卫星轨迹：生成每个时间步的ECEF坐标（米）和高度（千米）
    """
    all_traces = []
    total_steps = SIM_DURATION_SEC // TIME_STEP_SEC

    for step in range(total_steps):
        # 当前时间（秒级）
        current_sec = step * TIME_STEP_SEC
        current_time_ms = current_sec * MS_PER_SEC
        # 转换为Skyfield时间对象
        current_t = t0 + timedelta(seconds=current_sec)

        # 计算每颗卫星的坐标
        for sat_info in sat_metadata:
            sat = sat_info["satellite_obj"]
            # 获取地心坐标（GCRS惯性系），转换为ITRS地固系（ECEF）
            geocentric = sat.at(current_t)
            ecef_xyz_m = geocentric.frame_xyz(itrs).m  # 单位：米
            ecef_x, ecef_y, ecef_z = ecef_xyz_m

            # 计算高度（千米）
            subpoint = wgs84.subpoint(geocentric)
            altitude_km = subpoint.elevation.km

            # 组装轨迹数据（严格遵循项目文件格式）
            trace = {
                "time_ms": current_time_ms,
                "node_id": sat_info["node_id"],
                "name": sat_info["name"],
                "type": "SAT",
                "ecef_x": round(ecef_x, 2),
                "ecef_y": round(ecef_y, 2),
                "ecef_z": round(ecef_z, 2),
                "altitude_km": round(altitude_km, 2),
                "orbit_id": sat_info["orbit_id"],
                "ip": sat_info["ip"]
            }
            all_traces.append(trace)

    print(f"📊 完成 {total_steps} 个时间步的轨迹计算，共 {len(all_traces)} 条记录")
    return pd.DataFrame(all_traces)

def split_and_save_csv(trajectory_df):
    """
    按60秒切片保存CSV文件
    文件名格式：sat_trace_{startMs}_{endMs}.csv
    """
    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 计算切片数量
    total_chunks = SIM_DURATION_SEC // CHUNK_DURATION_SEC

    for chunk_idx in range(total_chunks):
        # 切片时间范围（毫秒）
        start_sec = chunk_idx * CHUNK_DURATION_SEC
        end_sec = start_sec + CHUNK_DURATION_SEC
        start_ms = start_sec * MS_PER_SEC
        end_ms = end_sec * MS_PER_SEC - 1  # 闭区间：[startMs, endMs]

        # 筛选当前切片的数据
        chunk_df = trajectory_df[
            (trajectory_df["time_ms"] >= start_ms) &
            (trajectory_df["time_ms"] < end_ms + 1)
        ]

        # 文件名
        filename = f"sat_trace_{start_ms}_{end_ms}.csv"
        file_path = os.path.join(OUTPUT_DIR, filename)

        # 保存CSV（不保留索引）
        chunk_df.to_csv(file_path, index=False, encoding="utf-8")
        print(f"💾 保存切片文件：{filename}（{len(chunk_df)} 条记录）")

    # 生成manifest.json（总索引文件）
    manifest = {
        "scenario_name": "rescue_mission_2026_v1",
        "t0_utc": T0_UTC.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_duration_sec": SIM_DURATION_SEC,
        "sat_count": MAX_SAT_COUNT,
        "trace_files": [
            f"sat_trace_{chunk_idx*CHUNK_DURATION_SEC*MS_PER_SEC}_"
            f"{(chunk_idx+1)*CHUNK_DURATION_SEC*MS_PER_SEC - 1}.csv"
            for chunk_idx in range(total_chunks)
        ]
    }
    manifest_path = os.path.join(
        os.path.dirname(OUTPUT_DIR), "manifest.json"
    )
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 生成索引文件：manifest.json")

def validate_trajectory_data(df):
    """
    数据校验：确保符合项目规范
    """
    print("\n🔍 开始数据校验...")
    valid = True

    # 1. 检查必填字段
    required_cols = ["time_ms", "node_id", "name", "type", "ecef_x", "ecef_y", "ecef_z", "altitude_km", "orbit_id", "ip"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必填字段：{missing_cols}")
        valid = False

    # 2. 检查时间戳连续性
    time_steps = sorted(df["time_ms"].unique())
    expected_steps = list(range(0, SIM_DURATION_SEC * MS_PER_SEC, TIME_STEP_SEC * MS_PER_SEC))
    if time_steps != expected_steps:
        print(f"❌ 时间戳不连续！期望 {len(expected_steps)} 个步骤，实际 {len(time_steps)} 个")
        valid = False

    # 3. 检查ECEF坐标合理性（地球半径~6371km，卫星高度~550km，总半径~6921km）
    earth_radius_km = 6371
    max_expected_radius_km = 7000  # 最大允许半径（避免卫星跑到外太空）
    df["radius_km"] = np.sqrt(
        (df["ecef_x"]/1000)**2 + (df["ecef_y"]/1000)**2 + (df["ecef_z"]/1000)**2
    )
    abnormal_radius = df[df["radius_km"] > max_expected_radius_km]
    if not abnormal_radius.empty:
        print(f"❌ 发现 {len(abnormal_radius)} 条异常坐标（半径超过 {max_expected_radius_km}km）")
        valid = False

    # 4. 检查空值
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"❌ 存在空值：{null_counts[null_counts > 0].to_dict()}")
        valid = False

    if valid:
        print("✅ 数据校验通过！所有规范均满足")
    else:
        raise ValueError("数据不符合项目规范，请检查配置或代码")

# ======================== 主流程 ========================
if __name__ == "__main__":

    '''
    if not download_latest_tle():
    # 下载失败且无本地文件时，终止程序
        exit(1)
    '''

    try:
        print("="*60)
        print("🚀 卫星轨迹生成程序（S1任务）启动")
        print(f"📅 仿真起始时间（UTC）：{T0_UTC.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  仿真时长：{SIM_DURATION_SEC}秒")
        print(f"📍 救援中心坐标：{OBS_LAT}°N, {OBS_LON}°E, {OBS_ELE}m")
        print("="*60)

        # 1. 初始化时间和观测点
        ts, t0 = init_time_scale()
        observer = Topos(
            latitude_degrees=OBS_LAT,
            longitude_degrees=OBS_LON,
            elevation_m=OBS_ELE
        )

        satellites = load.tle_file(TLE_FILE)
        all_starlink_sats = [sat for sat in satellites if "STARLINK" in sat.name.upper()]
        print(f"📡 预加载 {len(all_starlink_sats)} 颗Starlink卫星")

        trajectory_df = calculate_dynamic_sat_trajectory(all_starlink_sats, ts, t0, observer)
        validate_trajectory_data(trajectory_df)
        split_and_save_csv(trajectory_df)

        '''
        # 2. 筛选卫星并生成元数据
        sat_metadata = load_and_filter_satellites(t0, observer)

        # 3. 计算卫星轨迹
        trajectory_df = calculate_sat_trajectory(sat_metadata, ts, t0)

        # 4. 数据校验
        validate_trajectory_data(trajectory_df)

        # 5. 切片保存文件
        split_and_save_csv(trajectory_df)
        '''

        print("\n" + "="*60)
        print("🎉 卫星轨迹生成完成！")
        print(f"📁 输出目录：{OUTPUT_DIR}")
        print(f"📦 生成文件数：{SIM_DURATION_SEC // CHUNK_DURATION_SEC} 个CSV切片 + 1个manifest.json")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise
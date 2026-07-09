from skyfield.api import load, EarthSatellite, Topos, wgs84
from skyfield.framelib import itrs
from skyfield.nutationlib import iau2000b
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

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
IP_PREFIX = "10.0.3."  # 卫星IP前缀

# 4. 文件配置
# 获取当前代码文件的绝对路径（不受运行目录影响）
CODE_FILE_PATH = os.path.abspath(__file__)
# 获取代码文件所在的目录（路径基准）
CODE_DIR = os.path.dirname(CODE_FILE_PATH)
PARENT_DIR = os.path.dirname(CODE_DIR)
CELESTRAK_STARLINK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" # 星链TLE数据源
TLE_FILE = os.path.join(CODE_DIR, "starlink.tle") # 本地TLE文件路径
OUTPUT_BASE = os.path.join(
    PARENT_DIR,
    "S3",
    "traces"
)  # 输出根目录（S3模块用，子目录为 sat_trace_{N}）
VIS_SAT_TRACE_DIR = os.path.join(
    PARENT_DIR,
    "vis",
    "public",
    "data",
    "sat_trace"
)  # vis前端卫星轨迹目录
CHUNK_DURATION_SEC = 60  # 每个文件的时间切片（60秒）

# 5. 动态筛选配置
DYNAMIC_FILTER_INTERVAL_SEC = 60  # 动态筛选时间窗口（每60秒重新筛选一次）
RESELECT_SAT_COUNT = 40  # 每次动态筛选保留的卫星数（可被环境变量覆盖）

# 6. 性能优化配置
COARSE_FILTER_DIST_KM = 4000.0   # 粗筛选 3D 距离阈值（km），实际阈值 2000km，保守 2x
MP_BATCH_SIZE = 200              # 多进程每批卫星数
MP_MIN_SATS_FOR_PARALLEL = 1500  # 候选数少于此值不启用多进程，避免进程创建开销

# 地球自转速率（弧度/分钟，恒星日）
_EARTH_ROT_RAD_PER_MIN = 2.0 * np.pi / (23.0 * 60.0 + 56.0 + 4.0905 / 60.0)
# WGS84 椭球参数（粗筛选用）
_WGS84_A = 6378.137  # 长半轴 km
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F ** 2
# T0 的儒略日（粗筛选用）
_J2000 = datetime(2000, 1, 1, 12, 0, 0)
_JD_T0 = 2451545.0 + (T0_UTC - _J2000).total_seconds() / 86400.0
_GMST0_DEG = (280.46061837 + 360.98564736629 * (_JD_T0 - 2451545.0)) % 360
_GMST0_RAD = np.deg2rad(_GMST0_DEG)

# 从环境变量加载用户配置（前端传入，覆盖以上默认值）
_S1_CONFIG_FILE = os.environ.get('S1_CONFIG_FILE', '')
if _S1_CONFIG_FILE and os.path.exists(_S1_CONFIG_FILE):
    with open(_S1_CONFIG_FILE, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    if 'duration' in _cfg:     SIM_DURATION_SEC = int(_cfg['duration'])
    if 'obs_lat' in _cfg:      OBS_LAT = float(_cfg['obs_lat'])
    if 'obs_lon' in _cfg:      OBS_LON = float(_cfg['obs_lon'])
    if 'obs_ele' in _cfg:      OBS_ELE = float(_cfg['obs_ele'])
    if 'reselect_count' in _cfg: RESELECT_SAT_COUNT = int(_cfg['reselect_count'])
    if 'min_alt' in _cfg:      MIN_ALT_DEG = float(_cfg['min_alt'])
    if 'max_dist' in _cfg:     MAX_DIST_KM = float(_cfg['max_dist'])
    if 'chunk_duration' in _cfg: CHUNK_DURATION_SEC = int(_cfg['chunk_duration'])
    print(f"✅ 已加载用户配置: RESELECT={RESELECT_SAT_COUNT}, DUR={SIM_DURATION_SEC}s, OBS=({OBS_LAT},{OBS_LON},{OBS_ELE})")

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


# ======================== 性能优化：粗粒度预筛选 + 多进程 ========================

# ---- TLE 行缓存（用于多进程传递卫星信息，避免 pickle EarthSatellite） ----
_tle_line_cache = {}  # norad_id -> (name, line1, line2)


def _build_tle_line_cache(tle_file_path):
    """解析 TLE 文件，构建 NORAD ID → TLE 行映射（一次解析，多次复用）"""
    global _tle_line_cache
    with open(tle_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    _tle_line_cache.clear()
    for i in range(0, len(lines) - 2, 3):
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        # TLE line1 格式：列 3-7（1-indexed）为 NORAD ID
        try:
            norad_id = int(line1[2:7])
            _tle_line_cache[norad_id] = (name, line1, line2)
        except (ValueError, IndexError):
            continue
    print(f"📋 TLE 行缓存已构建：{len(_tle_line_cache)} 颗卫星")


# ---- 轨道数据缓存（用于粗筛选，避免重复提取轨道要素） ----
_orbit_data_cache = None


def _build_orbit_data(satellites):
    """
    从 Skyfield 卫星列表提取轨道要素到 NumPy 数组。
    只构建一次，后续每次粗筛选直接复用。
    """
    n = len(satellites)
    incl = np.empty(n, dtype=np.float64)
    node = np.empty(n, dtype=np.float64)
    argp = np.empty(n, dtype=np.float64)
    M0 = np.empty(n, dtype=np.float64)
    n_motion = np.empty(n, dtype=np.float64)  # mean motion, rad/min
    epoch_min = np.empty(n, dtype=np.float64)  # epoch offset from T0, minutes
    norad_ids = np.empty(n, dtype=np.int64)
    sat_objs = np.empty(n, dtype=object)

    for i, sat in enumerate(satellites):
        m = sat.model
        incl[i] = np.deg2rad(m.inclo)
        node[i] = np.deg2rad(m.nodeo)
        argp[i] = np.deg2rad(m.argpo)
        M0[i] = np.deg2rad(m.mo)
        n_motion[i] = m.no_kozai  # rad/min

        # 解析 TLE epoch
        yr = m.epochyr
        if yr < 57:
            yr += 2000
        else:
            yr += 1900
        doy = int(m.epochdays)
        frac_day = m.epochdays - doy
        epoch_dt = datetime(yr, 1, 1) + timedelta(
            days=doy - 1, seconds=frac_day * 86400
        )
        epoch_min[i] = (epoch_dt - T0_UTC).total_seconds() / 60.0

        norad_ids[i] = m.satnum
        sat_objs[i] = sat

    # 观测点 ECEF 坐标（在此时计算以反映可能的配置覆盖）
    obs_lat_rad = np.deg2rad(OBS_LAT)
    obs_lon_rad = np.deg2rad(OBS_LON)
    n_sin = np.sqrt(1.0 - _WGS84_E2 * np.sin(obs_lat_rad) ** 2)
    n_val = _WGS84_A / n_sin
    obs_ecef_x = (n_val + OBS_ELE / 1000.0) * np.cos(obs_lat_rad) * np.cos(obs_lon_rad)
    obs_ecef_y = (n_val + OBS_ELE / 1000.0) * np.cos(obs_lat_rad) * np.sin(obs_lon_rad)
    obs_ecef_z = (n_val * (1.0 - _WGS84_E2) + OBS_ELE / 1000.0) * np.sin(obs_lat_rad)

    return {
        'incl': incl, 'node': node, 'argp': argp,
        'M0': M0, 'n_motion': n_motion, 'epoch_min': epoch_min,
        'norad_ids': norad_ids, 'sat_objs': sat_objs,
        'obs_ecef_x': obs_ecef_x, 'obs_ecef_y': obs_ecef_y, 'obs_ecef_z': obs_ecef_z,
    }


def _coarse_filter_candidates(orbit_data, elapsed_minutes):
    """
    矢量化粗粒度卫星可见性筛选（单次调用 ~10ms，覆盖上万颗卫星）。

    使用简化的圆轨道传播器 + 3D ECEF 坐标变换，
    快速估算每颗卫星到观测点的直线距离。
    保守设计：宁可保留不可见卫星（后续精确筛选排除），绝不漏选。

    返回：候选卫星在原始列表中的索引数组。
    """
    incl = orbit_data['incl']
    node = orbit_data['node']
    argp = orbit_data['argp']
    M0 = orbit_data['M0']
    n_motion = orbit_data['n_motion']
    epoch_min = orbit_data['epoch_min']

    # ── 简化轨道传播（圆轨道近似，e≈0） ──
    # 当前平近点角: M = M0 + n * (t - t_epoch)
    M = M0 + n_motion * (elapsed_minutes - epoch_min)
    # 纬度幅角（近圆轨道：真近点角 ≈ 平近点角）
    u = argp + M

    # 卫星轨道半径（近似常数：地球半径 + 550km Starlink 典型高度）
    R_sat = 6371.0 + 550.0  # km

    # ── ECI 坐标（地心惯性系） ──
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_incl = np.cos(incl)
    sin_incl = np.sin(incl)
    cos_node = np.cos(node)
    sin_node = np.sin(node)

    # 轨道面 → ECI 旋转
    x_eci = R_sat * (cos_u * cos_node - sin_u * cos_incl * sin_node)
    y_eci = R_sat * (cos_u * sin_node + sin_u * cos_incl * cos_node)
    z_eci = R_sat * sin_u * sin_incl

    # ── ECI → ECEF（绕 Z 轴旋转 -GMST） ──
    # GMST(t) = GMST0 + ω_earth * elapsed_minutes
    gmst = _GMST0_RAD + _EARTH_ROT_RAD_PER_MIN * elapsed_minutes
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)

    x_ecef = x_eci * cos_gmst + y_eci * sin_gmst
    y_ecef = -x_eci * sin_gmst + y_eci * cos_gmst
    z_ecef = z_eci

    # ── 到观测点的 3D 直线距离（保守阈值 4000km >> 可见最大 ~2704km） ──
    dx = x_ecef - orbit_data['obs_ecef_x']
    dy = y_ecef - orbit_data['obs_ecef_y']
    dz = z_ecef - orbit_data['obs_ecef_z']
    dist_km = np.sqrt(dx * dx + dy * dy + dz * dz)

    candidate_mask = dist_km < COARSE_FILTER_DIST_KM
    return np.where(candidate_mask)[0]


# ---- 多进程精确筛选 ----

# Worker 进程全局变量（通过 initializer 设置，每个子进程独立）
_worker_sats = None
_worker_ts = None


def _mp_worker_init(candidate_tle_list):
    """
    多进程 worker 初始化：预加载本进程需要检查的候选卫星。
    每个子进程只调用一次，后续 _mp_check_batch 复用。
    """
    global _worker_sats, _worker_ts
    _worker_ts = load.timescale()
    _worker_sats = {}
    for norad_id, name, line1, line2 in candidate_tle_list:
        try:
            _worker_sats[norad_id] = EarthSatellite(line1, line2, name, _worker_ts)
        except Exception:
            pass  # 个别卫星 TLE 解析失败则跳过


def _mp_check_batch(args):
    """
    多进程 batch 检查：检查一批卫星在当前时刻的可见性。
    在 worker 进程中调用，通过全局变量访问预构建的卫星对象。
    """
    global _worker_sats, _worker_ts

    norad_ids, t_jd, obs_lat, obs_lon, obs_ele, min_alt, max_dist = args

    t = _worker_ts.tt(jd=t_jd)
    observer = Topos(
        latitude_degrees=obs_lat,
        longitude_degrees=obs_lon,
        elevation_m=obs_ele,
    )

    results = []
    for nid in norad_ids:
        sat = _worker_sats.get(nid)
        if sat is None:
            continue
        try:
            diff = sat - observer
            topo = diff.at(t)
            alt = topo.altaz()[0].degrees
            dist = topo.distance().km
            if alt > min_alt or dist < max_dist:
                results.append((dist, nid))
        except Exception:
            pass  # 个别卫星计算失败则跳过

    return results


def filter_visible_satellites_optimized(all_starlink_sats, observer, current_t,
                                         orbit_data, elapsed_minutes):
    """
    优化版卫星筛选：粗粒度预筛选 →（候选多时）多进程精确检查。
    与 filter_visible_satellites 保持相同的输入输出接口。
    """
    # ── Step 1: 粗粒度预筛选（矢量化 NumPy，数毫秒完成） ──
    candidate_indices = _coarse_filter_candidates(
        orbit_data, elapsed_minutes,
    )

    n_candidates = len(candidate_indices)

    # 安全回退：粗筛选结果太少则使用全量卫星（避免边缘情况漏选）
    if n_candidates < RESELECT_SAT_COUNT:
        print(f"  ⚠️ 粗筛选仅得 {n_candidates} 颗（<{RESELECT_SAT_COUNT}），回退到全量筛选")
        candidate_indices = np.arange(len(all_starlink_sats))

    # 构建候选卫星列表（主进程的 EarthSatellite 对象）
    candidates = [all_starlink_sats[i] for i in candidate_indices]
    n_candidates = len(candidates)

    # ── Step 2: 精确可见性检查 ──
    t_jd = current_t.tt

    if n_candidates >= MP_MIN_SATS_FOR_PARALLEL:
        # ── 多进程并行路径 ──
        n_workers = max(1, mp.cpu_count() - 1)

        # 从 TLE 缓存中提取候选卫星的 TLE 行（避免 pickle EarthSatellite）
        candidate_tle_list = []
        for idx in candidate_indices:
            sat = all_starlink_sats[idx]
            nid = sat.model.satnum
            tle_info = _tle_line_cache.get(nid)
            if tle_info:
                candidate_tle_list.append(
                    (nid, tle_info[0], tle_info[1], tle_info[2])
                )

        # 将候选卫星分批
        batches = []
        for i in range(0, len(candidate_indices), MP_BATCH_SIZE):
            batch_ids = [
                int(all_starlink_sats[idx].model.satnum)
                for idx in candidate_indices[i:i + MP_BATCH_SIZE]
            ]
            batches.append((
                batch_ids, t_jd,
                OBS_LAT, OBS_LON, OBS_ELE,
                MIN_ALT_DEG, MAX_DIST_KM,
            ))

        # 并行执行
        all_visible = []
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_mp_worker_init,
            initargs=(candidate_tle_list,),
        ) as executor:
            for result in executor.map(_mp_check_batch, batches):
                all_visible.extend(result)

        print(f"  ✅ 并行筛选完成：{n_candidates} 颗候选 → {len(all_visible)} 颗可见 "
              f"（{n_workers} workers, {len(batches)} batches）")
    else:
        # ── 串行路径（候选少时避免多进程创建开销） ──
        all_visible = []
        for sat in candidates:
            try:
                diff = sat - observer
                topo = diff.at(current_t)
                alt = topo.altaz()[0].degrees
                dist = topo.distance().km
                if alt > MIN_ALT_DEG or dist < MAX_DIST_KM:
                    all_visible.append((dist, sat.model.satnum))
            except Exception:
                pass

    # ── Step 3: 排序取前 N 颗 ──
    all_visible.sort(key=lambda x: x[0])
    selected = all_visible[:RESELECT_SAT_COUNT]

    # ── Step 4: 构建元数据（使用主进程的卫星对象，确保与原逻辑一致） ──
    sat_lookup = {sat.model.satnum: sat for sat in all_starlink_sats}
    sat_metadata = []
    for idx, (dist_km, norad_id) in enumerate(selected, 1):
        sat = sat_lookup.get(norad_id)
        if sat is None:
            continue
        sat_id = f"SAT_{norad_id:05d}"
        ip = f"{IP_PREFIX}{norad_id % 255}"
        sat_metadata.append({
            "node_id": sat_id,
            "name": sat.name.strip(),
            "ip": ip,
            "orbit_id": -1,
            "satellite_obj": sat,
            "norad_id": norad_id,
            "current_dist_km": round(dist_km, 2),
        })
    return sat_metadata


def calculate_dynamic_sat_trajectory(all_starlink_sats, ts, t0, observer):
    """
    动态计算卫星轨迹：每DYNAMIC_FILTER_INTERVAL_SEC秒重新筛选可见卫星。
    使用粗粒度预筛选 + 多进程并行筛选加速（优化版）。
    """
    all_traces = []
    total_steps = SIM_DURATION_SEC // TIME_STEP_SEC

    # 构建轨道数据缓存（一次性，后续筛选复用）
    print(f"🔧 构建轨道数据缓存（{len(all_starlink_sats)} 颗卫星）...")
    orbit_data = _build_orbit_data(all_starlink_sats)
    print(f"📡 预加载 {len(all_starlink_sats)} 颗Starlink卫星，开始动态轨迹计算...")

    for step in range(total_steps):
        current_sec = step * TIME_STEP_SEC
        current_time_ms = current_sec * MS_PER_SEC
        current_t = t0 + timedelta(seconds=current_sec)

        # 每N秒重新筛选一次可见卫星（使用优化版筛选）
        if current_sec % DYNAMIC_FILTER_INTERVAL_SEC == 0:
            elapsed_minutes = current_sec / 60.0
            current_sat_metadata = filter_visible_satellites_optimized(
                all_starlink_sats, observer, current_t,
                orbit_data, elapsed_minutes,
            )
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

def split_and_save_csv(trajectory_df, output_dir):
    """按60秒切片保存CSV文件"""
    os.makedirs(output_dir, exist_ok=True)

    total_chunks = SIM_DURATION_SEC // CHUNK_DURATION_SEC

    for chunk_idx in range(total_chunks):
        start_sec = chunk_idx * CHUNK_DURATION_SEC
        end_sec = start_sec + CHUNK_DURATION_SEC
        start_ms = start_sec * MS_PER_SEC
        end_ms = end_sec * MS_PER_SEC - 1

        chunk_df = trajectory_df[
            (trajectory_df["time_ms"] >= start_ms) &
            (trajectory_df["time_ms"] < end_ms + 1)
        ]

        filename = f"sat_trace_{start_ms}_{end_ms}_{RESELECT_SAT_COUNT}.csv"
        file_path = os.path.join(output_dir, filename)

        chunk_df.to_csv(file_path, index=False, encoding="utf-8")
        print(f"💾 保存切片文件：{filename}（{len(chunk_df)} 条记录）")

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

        # 性能优化：构建 TLE 行缓存（用于多进程传递卫星信息）
        _build_tle_line_cache(TLE_FILE)

        trajectory_df = calculate_dynamic_sat_trajectory(all_starlink_sats, ts, t0, observer)
        validate_trajectory_data(trajectory_df)
        # 输出到 S3 模块
        output_dir = os.path.join(OUTPUT_BASE, f"sat_trace_{RESELECT_SAT_COUNT}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 S3 输出目录：{output_dir}")
        split_and_save_csv(trajectory_df, output_dir)

        # 同时输出到 vis 前端
        vis_dir = VIS_SAT_TRACE_DIR
        os.makedirs(vis_dir, exist_ok=True)
        print(f"📁 vis 前端输出目录：{vis_dir}")
        split_and_save_csv(trajectory_df, vis_dir)

        print("\n" + "="*60)
        print("🎉 卫星轨迹生成完成！")
        print(f"📁 S3 输出目录：{output_dir}")
        print(f"📁 vis 前端目录：{vis_dir}")
        print(f"📦 生成文件数：{SIM_DURATION_SEC // CHUNK_DURATION_SEC} 个CSV切片 × 2 处 + 1个manifest.json")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise
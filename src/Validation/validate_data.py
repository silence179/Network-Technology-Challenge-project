import pandas as pd
import numpy as np
import os
import re
from pymap3d.ecef import ecef2geodetic

# S1 卫星(SAT) 专属校验函数
# 适配表头：time_ms,node_id,name,type,ecef_x,ecef_y,ecef_z,altitude_km,orbit_id,ip,radius_km
# 适配规则：50个SAT为一组，同time_ms对应50行，time_ms 1000ms递增（1Hz）
def validate_s1_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        row_count = len(df)
        file_name = os.path.basename(file_path)

        # 基础格式校验：文件名+表头完整性
        if not re.match(r"^sat_trace_\d+_\d+\.csv$", file_name):
            return "FAIL", f"文件名格式错误，必须为sat_trace_{startMs}_{endMs}.csv"
        S1_REQUIRED_COLS = ["time_ms", "node_id", "name", "type", "ecef_x", 
                            "ecef_y", "ecef_z", "altitude_km", "orbit_id", "ip", "radius_km"]
        missing_cols = [col for col in S1_REQUIRED_COLS if col not in df.columns]
        if missing_cols:
            return "FAIL", f"表头缺失必填字段：{','.join(missing_cols)}"

        # 空值兜底校验：所有字段无空值
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            return "FAIL", f"CSV存在空值，空值字段：{','.join(null_cols)}"

        # 50行一组格式校验
        time_group_size = df.groupby("time_ms").size()
        abnormal_group = time_group_size[time_group_size != 50]
        if not abnormal_group.empty:
            return "FAIL", f"时间戳分组异常，以下time_ms行数非50行：{abnormal_group.index.tolist()}"
        duplicate_node = df[df.duplicated(subset=["time_ms", "node_id"])]
        if not duplicate_node.empty:
            return "FAIL", f"同时间戳内node_id重复：{duplicate_node[['time_ms','node_id']].head(3).values.tolist()}"

        # 时间戳连续性校验：1000ms步长，60秒切片60个唯一值
        unique_time = np.sort(df["time_ms"].unique())
        if len(unique_time) == 0:
            return "FAIL", "无有效time_ms数据"
        time_step = np.diff(unique_time)
        if not np.all(time_step == 1000):
            abnormal_steps = time_step[time_step != 1000].unique()
            return "FAIL", f"time_ms递增异常，应1000ms步长，异常步长：{abnormal_steps}ms"
        if len(unique_time) != 60:
            return "FAIL", f"60秒切片唯一time_ms数量异常，应60个，实际{len(unique_time)}个"

        # SAT专属属性校验
        if not df["type"].eq("SAT").all():
            abnormal_type = df["type"].unique()
            return "FAIL", f"type字段异常，必须全为SAT，当前包含：{abnormal_type}"
        abnormal_alt = df[(df["altitude_km"] < 200) | (df["altitude_km"] > 1200)]
        if not abnormal_alt.empty:
            return "FAIL", f"卫星高度异常（200-1200km）：{abnormal_alt[['node_id','time_ms','altitude_km']].head(3).values.tolist()}"
        abnormal_radius = df[(df["radius_km"] < 6371) | (df["radius_km"] > 7000)]
        if not abnormal_radius.empty:
            return "FAIL", f"ECEF半径异常（6371-7000km）：{abnormal_radius[['node_id','time_ms','radius_km']].head(3).values.tolist()}"

        # IP格式合规校验
        ip_pattern = r"^10\.0\.3\.\d{1,3}$"
        invalid_ip = df[~df["ip"].str.match(ip_pattern, na=False)]
        if not invalid_ip.empty:
            return "FAIL", f"IP格式异常（必须10.0.3.x），异常IP：{invalid_ip['ip'].unique()[:3]}"

        return "PASS", "所有S1(SAT)校验项通过，格式+属性均合规"

    except Exception as e:
        return "ERROR", f"文件读取/校验异常：{str(e)[:100]}"

# S2 空地(GS+UAV) 专属校验函数
def validate_s2_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        row_count = len(df)
        file_name = os.path.basename(file_path)

        # 基础格式校验：文件名+表头+空值
        if not re.match(r"^uav_trace_\d+_\d+\.csv$", file_name):
            return "FAIL", f"文件名格式错误，必须为uav_trace_{startMs}_{endMs}.csv"
        S2_REQUIRED_COLS = ["time_ms", "node_id", "role", "type", "ecef_x", "ecef_y", 
                            "ecef_z", "ip", "heading_deg", "battery_pct"]
        missing_cols = [col for col in S2_REQUIRED_COLS if col not in df.columns]
        if missing_cols:
            return "FAIL", f"表头缺失必填字段：{','.join(missing_cols)}"
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            return "FAIL", f"CSV存在空值，空值字段：{','.join(null_cols)}"

        # 4行一组格式校验：1GS+3UAV，同time_ms对应4行
        time_group_size = df.groupby("time_ms").size()
        abnormal_group = time_group_size[time_group_size != 4]
        if not abnormal_group.empty:
            return "FAIL", f"时间戳分组异常，以下time_ms行数非4行：{abnormal_group.index.tolist()}"
        duplicate_node = df[df.duplicated(subset=["time_ms", "node_id"])]
        if not duplicate_node.empty:
            return "FAIL", f"同时间戳内node_id重复：{duplicate_node[['time_ms','node_id']].head(3).values.tolist()}"

        # 时间戳连续性：100ms步长，60秒切片600个唯一值，总2400行
        unique_time = np.sort(df["time_ms"].unique())
        if len(unique_time) != 600:
            return "FAIL", f"60秒切片唯一time_ms数量异常，应600个，实际{len(unique_time)}个"
        time_step = np.diff(unique_time)
        if not np.all(time_step == 100):
            abnormal_steps = time_step[time_step != 100].unique()
            return "FAIL", f"time_ms递增异常，应100ms步长，异常步长：{abnormal_steps}ms"
        if row_count != 2400:
            return "FAIL", f"60秒切片（4行/组）总行列异常，应2400行，实际{row_count}行"

        # 坐标校验：ECEF模长+四川锚点经纬度+高度
        try:
            lat, lon, alt = ecef2geodetic(df["ecef_x"].values, df["ecef_y"].values, df["ecef_z"].values)
            df["lat"] = lat
            df["lon"] = lon
            df["alt_m"] = alt
            df["ecef_mag_km"] = np.sqrt(df["ecef_x"]**2 + df["ecef_y"]**2 + df["ecef_z"]**2) / 1000

            # ECEF模长合理性校验
            ecef_min = 6371
            ecef_max = 7000
            abnormal_ecef = df[(df["ecef_mag_km"] < ecef_min) | (df["ecef_mag_km"] > ecef_max)]
            if not abnormal_ecef.empty:
                err_data = abnormal_ecef[['node_id','time_ms','ecef_mag_km']].head(3).values.tolist()
                return "FAIL", f"ECEF模长异常（需{ecef_min}-{ecef_max}km）：{err_data}"

            # 经纬度合理性校验
            lat_min, lat_max = 29.0, 31.0
            lon_min, lon_max = 103.0, 105.0
            abnormal_pos = df[(df["lat"] < lat_min) | (df["lat"] > lat_max) | 
                            (df["lon"] < lon_min) | (df["lon"] > lon_max)]
            if not abnormal_pos.empty:
                err_data = abnormal_pos[['node_id','time_ms','lat','lon']].head(3).values.tolist()
                return "FAIL", f"经纬度越界（需{lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E）：{err_data}"

            # 打印GS/UAV海拔用于问题定位
            uav_mask = df["type"] == "UAV"
            gs_mask = df["type"] == "GS"
            if not df[gs_mask].empty:
                print(f"【GS真实海拔】：{df[gs_mask]['alt_m'].unique()} 米")
            if not df[uav_mask].empty:
                print(f"【UAV真实海拔】：{df[uav_mask]['alt_m'].unique()[:3]} 米")
            
            # 高度合理性校验（按类型差异化）
            uav_alt_min, uav_alt_max = 500.0, 5000.0
            abnormal_uav_alt = df[uav_mask & ((df["alt_m"] < uav_alt_min) | (df["alt_m"] > uav_alt_max))]
            if not abnormal_uav_alt.empty:
                err_data = abnormal_uav_alt[['node_id','time_ms','alt_m']].head(3).values.tolist()
                return "FAIL", f"UAV高度异常（需{uav_alt_min}-{uav_alt_max}m）：{err_data}"

            # GS高度校验：≤1000m且无漂移
            gs_alt_max = 1000.0
            if not df[gs_mask].empty:
                abnormal_gs_alt = df[gs_mask & (df["alt_m"] > gs_alt_max)]
                if not abnormal_gs_alt.empty:
                    err_data = abnormal_gs_alt[['node_id','time_ms','alt_m']].head(3).values.tolist()
                    return "FAIL", f"GS海拔异常（地面站需≤{gs_alt_max}m）：{err_data}"
                gs_alt_unique = df[gs_mask]["alt_m"].unique()
                if len(gs_alt_unique) > 1:
                    return "FAIL", f"GS海拔漂移（所有时间戳需一致）：{gs_alt_unique}"

            # 坐标值有效性兜底校验
            invalid_coord = df[df[["ecef_x","ecef_y","ecef_z"]].isin([np.inf, -np.inf, np.nan]).any(axis=1)]
            if not invalid_coord.empty:
                err_data = invalid_coord[['node_id','time_ms']].head(3).values.tolist()
                return "FAIL", f"坐标值无效（含无穷大/空值）：{err_data}"

        except Exception as e:
            return "FAIL", f"坐标校验异常：{str(e)[:100]}"

        # 动态属性校验：航向角+电池+角色
        # 航向角校验
        if not df[gs_mask]["heading_deg"].eq(-1).all():
            return "FAIL", f"GS的heading_deg必须全为-1，发现异常值"
        if uav_mask.any():
            df.loc[uav_mask, "heading_deg"] = df.loc[uav_mask, "heading_deg"] % 360
            uav_abnormal_heading = df[uav_mask & ((df["heading_deg"] < 0) | (df["heading_deg"] > 360))]
            if not uav_abnormal_heading.empty:
                return "FAIL", f"UAV航向角异常（0-360°）：{uav_abnormal_heading[['node_id','heading_deg']].head(3).values.tolist()}"
        
        # 电池校验：UAV变化率≤10%/秒，GS=-1
        if not df[gs_mask]["battery_pct"].eq(-1).all():
            return "FAIL", f"GS的battery_pct必须全为-1，发现异常值"
        uav_df = df[uav_mask].copy()
        for uid in uav_df["node_id"].unique():
            single_uav = uav_df[uav_df["node_id"] == uid].sort_values("time_ms")
            battery_diff = np.diff(single_uav["battery_pct"])
            time_diff = np.diff(single_uav["time_ms"]) / 1000
            valid_idx = time_diff != 0
            if valid_idx.any():
                change_rate = battery_diff[valid_idx] / time_diff[valid_idx]
                if (np.abs(change_rate) > 10).any():
                    abnormal_rate = change_rate[np.abs(change_rate) > 10].round(2)
                    return "FAIL", f"UAV {uid} 电池突变（变化率>10%/秒），异常变化率：{abnormal_rate[:5]}"
        
        # 角色校验：UAV移动（>5米）时必须为RELAY
        MOVE_THRESHOLD = 5
        for uid in uav_df["node_id"].unique():
            single_uav = df[df["node_id"] == uid].sort_values("time_ms")
            pos_diff = np.sqrt(np.diff(single_uav["ecef_x"])**2 + np.diff(single_uav["ecef_y"])**2 + np.diff(single_uav["ecef_z"])**2)
            moving = pos_diff > MOVE_THRESHOLD
            roles = single_uav["role"].values[1:]
            invalid_role = roles[moving & (roles != "RELAY")]
            if len(invalid_role) > 0:
                return "FAIL", f"UAV {uid} 移动时角色不为RELAY"

        # IP格式校验
        ip_pattern = r"^10\.0\.0\.\d{1,3}$"
        invalid_ip = df[~df["ip"].str.match(ip_pattern, na=False)]
        if not invalid_ip.empty:
            return "FAIL", f"IP格式异常（必须10.0.0.x），异常IP：{invalid_ip['ip'].unique()[:3]}"

        return "PASS", "所有S2(GS+UAV)校验项通过，格式+属性均合规"

    except Exception as e:
        return "ERROR", f"文件读取/校验异常：{str(e)[:100]}"

# 批量校验主函数：自动区分S1/S2文件类型
def batch_validate (csv_dir="../data/scenarios/rescue_mission_2026_v1/traces"):
    if not os.path.exists(csv_dir):
        print(f"❌ 错误：未找到数据文件夹 {csv_dir}")
        return
    csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"⚠️  提示：{csv_dir} 文件夹下无CSV文件")
        return
    for file in csv_files:
        file_path = os.path.join(csv_dir, file)
        if file.startswith("sat_trace_"):
            status, msg = validate_s1_csv(file_path)
            print(f"【S1-SAT | {file}】→ {status}：{msg}")
        elif file.startswith("uav_trace_"):
            status, msg = validate_s2_csv(file_path)
            print(f"【S2-GS+UAV | {file}】→ {status}：{msg}")
        else:
            print(f"【未知类型 | {file}】→ 跳过：文件名需以sat_trace_/uav_trace_开头")

# 程序运行入口
if __name__ == "__main__":
    batch_validate()
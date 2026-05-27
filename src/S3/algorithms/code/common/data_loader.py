"""
Data loader: load satellite/UAV traces from CSV files.
"""

import os
import glob
import pandas as pd

from ..config import SAT_DIR, UAV_FILE


def _resolve_uav_file(uav_file):
    candidates = [uav_file]
    basename = os.path.basename(uav_file)
    parent_dir = os.path.dirname(uav_file)
    nested_candidate = os.path.join(parent_dir, 'uav_trace', basename)
    flat_candidate = os.path.join(parent_dir, basename)

    for candidate in (nested_candidate, flat_candidate):
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return uav_file


def load_traces(sat_dir=SAT_DIR, uav_file=UAV_FILE):
    """Load satellite and UAV trace DataFrames and extract timestamps."""
    print(">>> Loading traces...")
    sat_files = glob.glob(os.path.join(sat_dir, "*.csv"))
    df_sat = pd.concat([pd.read_csv(f) for f in sat_files], ignore_index=True) if sat_files else pd.DataFrame()
    resolved_uav_file = _resolve_uav_file(uav_file)
    df_uav = pd.read_csv(resolved_uav_file) if os.path.exists(resolved_uav_file) else pd.DataFrame()
    timestamps = sorted(df_uav['time_ms'].unique()) if not df_uav.empty else []
    print(f"    SAT files: {len(sat_files)}, time steps: {len(timestamps)}")
    return df_sat, df_uav, timestamps


def get_nodes(df_sat, df_uav, t_ms):
    """Get node positions at a given timestamp."""
    uav_t = df_uav[df_uav['time_ms'] == t_ms]
    sat_key = (t_ms // 1000) * 1000
    sat_t = df_sat[df_sat['time_ms'] == sat_key]
    cols = ['node_id', 'type', 'ecef_x', 'ecef_y', 'ecef_z', 'ip']
    if sat_t.empty and uav_t.empty:
        return pd.DataFrame(columns=cols)
    return pd.concat([sat_t[cols], uav_t[cols]], ignore_index=True)

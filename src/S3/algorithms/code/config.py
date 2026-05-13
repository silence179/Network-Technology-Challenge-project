"""
Global configuration for OLCP experiments.
All configurable parameters are defined here.
"""

import os
import random
import numpy as np

# ── Paths ──
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(CODE_DIR, '..', '..', 'traces')
SAT_DIR = os.path.join(TRACES_DIR, 'sat_trace_100')


def _resolve_default_uav_file():
	preferred = os.path.join(TRACES_DIR, 'uav_trace', 'uav_trace_full.csv')
	fallback = os.path.join(TRACES_DIR, 'uav_trace_full.csv')
	return preferred if os.path.exists(preferred) else fallback


UAV_FILE = _resolve_default_uav_file()

# ── Network ──
MAX_LINK_RANGE = 5000 * 1000       # 5000 km in meters
MIN_ELEVATION = 10.0                # degrees
SPEED_OF_LIGHT = 3e8               # m/s

# ── Content ──
CONTENT_SIZE_MB = 10.0
CONTENT_SIZE_BITS = CONTENT_SIZE_MB * 8 * 1e6
CONTENT_CATALOG_SIZE = 100

# ── Cache ──
CACHE_SAT_COUNT = 3
CACHE_CAPACITY = 8
CACHE_SERVE_BW_MBPS = 35.0
GS_SERVE_BW_MBPS = 20.0
ORIGIN_SERVER = 'GS_01'

# ── Simulation ──
REQUESTS_PER_STEP = 16
MAX_STEPS = 20
STEP_STRIDE = 60          # sample every 60 timestamps (= 6s for 100ms UAV trace)
ZIPF_ALPHA = 1.2

# ── OLCP ──
LOOKAHEAD_HORIZON = 5              # H: planning horizon (steps)
DISCOUNT_FACTOR = 0.9              # γ: discount for future steps
MIGRATION_BUDGET = 4              # tuned default to keep the study out of saturation

# ── Packet Loss (ARQ retransmission model) ──
ENABLE_PACKET_LOSS = True
LOSS_RATE_ISL = 0.01            # SAT-SAT inter-satellite link: 1%
LOSS_RATE_SAT_UAV = 0.05        # SAT-UAV link: 5% (atmospheric fading)
LOSS_RATE_SAT_GS = 0.03         # SAT-GS link: 3%
LOSS_RATE_DEFAULT = 0.02         # fallback

# ── Popularity tracking ──
EWMA_DECAY = 0.95

# ── Baselines ──
GREEDY_REFRESH_INTERVAL = 5        # Greedy-Popular re-fills every N steps
LCE_LRU_CAPACITY = 8               # per-node LRU capacity

# ── Random seed ──
random.seed(42)
np.random.seed(42)

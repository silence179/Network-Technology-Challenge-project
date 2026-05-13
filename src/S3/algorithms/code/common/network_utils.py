"""
Network utility functions: distance, delay, bandwidth, elevation, packet loss.
"""

import math
import numpy as np

from ..config import (
    SPEED_OF_LIGHT, ENABLE_PACKET_LOSS,
    LOSS_RATE_ISL, LOSS_RATE_SAT_UAV, LOSS_RATE_SAT_GS, LOSS_RATE_DEFAULT,
)


def ecef_distance(a, b):
    """Euclidean distance between two ECEF coordinate tuples."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def propagation_delay_ms(dist_m):
    """Propagation delay in milliseconds for a given distance in meters."""
    return (dist_m / SPEED_OF_LIGHT) * 1000.0


def link_bandwidth_mbps(type_a, type_b):
    """Return link bandwidth (Mbps) based on node types."""
    types = {type_a, type_b}
    if 'GS' in types and 'UAV' in types:
        return 0
    if 'UAV' in types and 'SAT' in types:
        return 20.0
    if 'SAT' in types and 'GS' in types:
        return 20.0
    if types == {'SAT'}:
        return 100.0
    return 10.0


def elevation_deg(pos_gnd, pos_sat):
    """Elevation angle (degrees) from ground position to satellite."""
    vg = np.array(pos_gnd)
    vs = np.array(pos_sat) - vg
    dg = np.linalg.norm(vg)
    ds = np.linalg.norm(vs)
    if dg == 0 or ds == 0:
        return 90.0
    cos_t = np.clip(np.dot(vg, vs) / (dg * ds), -1.0, 1.0)
    return 90.0 - math.degrees(np.arccos(cos_t))


def packet_loss_rate(type_a, type_b):
    """Return per-link packet loss rate based on node types.

    Used for ARQ retransmission model: effective_delay = delay / (1 - loss).
    """
    if not ENABLE_PACKET_LOSS:
        return 0.0
    types = {type_a, type_b}
    if types == {'SAT'}:
        return LOSS_RATE_ISL
    if 'UAV' in types and 'SAT' in types:
        return LOSS_RATE_SAT_UAV
    if 'SAT' in types and 'GS' in types:
        return LOSS_RATE_SAT_GS
    return LOSS_RATE_DEFAULT


def effective_delay_ms(delay_ms, loss):
    """Effective delay with ARQ retransmission: delay / (1 - loss)."""
    if loss >= 1.0:
        return float('inf')
    return delay_ms / (1.0 - loss)

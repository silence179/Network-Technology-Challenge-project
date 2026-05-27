"""
OLCP LP Solver: Orbit-Lookahead Content Placement via Linear Programming.
Uses SPARSE matrices for scalability.

Formulation:
  max  Σ_τ γ^τ Σ_f λ_f Σ_c δ_c^(τ) · x_{c,f}^{(τ)}
  s.t. capacity, no-redundancy, migration budget, node availability
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csc_matrix
import networkx as nx

from .. import config as cfg
from ..config import (
    CONTENT_CATALOG_SIZE, CONTENT_SIZE_BITS,
    CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS,
)


def _compute_per_node_delays(G, cache_nodes_list, type_map):
    """Compute per-node delay savings δ_c and origin miss delay d_miss."""
    uav_nodes = [n for n, t in type_map.items() if t == 'UAV' and G.has_node(n)]
    if not uav_nodes:
        return {}, 0.0

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    miss_delays = []
    for u in uav_nodes:
        best = float('inf')
        for gs in gs_nodes:
            try:
                d = nx.dijkstra_path_length(G, u, gs, weight='eff_delay')
                best = min(best, d)
            except nx.NetworkXNoPath:
                pass
        if best < float('inf'):
            miss_delays.append(2 * best + transfer_origin)
    d_miss = np.mean(miss_delays) if miss_delays else 10000.0

    delta_c = {}
    for c in cache_nodes_list:
        if not G.has_node(c):
            continue
        hit_delays = []
        for u in uav_nodes:
            try:
                d = nx.dijkstra_path_length(G, u, c, weight='eff_delay')
                hit_delays.append(d + transfer_cache)
            except nx.NetworkXNoPath:
                pass
        if hit_delays:
            delta_c[c] = max(d_miss - np.mean(hit_delays), 0.0)
        else:
            delta_c[c] = 0.0

    return delta_c, d_miss


def solve_olcp(future_snapshots, current_cache_state, popularity_scores, horizon=None):
    """Solve the OLCP LP with sparse constraint matrices.

    Parameters
    ----------
    future_snapshots : list of dict
        Each: {'G': nx.Graph, 'cache_nodes': set/list, 'type_map': dict}
    current_cache_state : dict {node: set(content_ids)}
    popularity_scores : dict {content_id: float}
    horizon : int or None

    Returns
    -------
    placement : dict {node: set(content_ids)}
    lp_value : float
    """
    H = horizon if horizon is not None else len(future_snapshots) - 1
    H = min(H, len(future_snapshots) - 1)
    if H < 0:
        return current_cache_state.copy(), 0.0

    F = CONTENT_CATALOG_SIZE
    gamma = cfg.DISCOUNT_FACTOR
    C_cap = cfg.CACHE_CAPACITY
    M_bud = cfg.MIGRATION_BUDGET

    # Collect all distinct cache nodes
    all_cache_nodes = set()
    cache_nodes_per_step = []
    delta_per_step = []

    for snap in future_snapshots[:H + 1]:
        nodes = list(snap['cache_nodes'])
        cache_nodes_per_step.append(nodes)
        all_cache_nodes.update(nodes)
        delta_c, _ = _compute_per_node_delays(snap['G'], nodes, snap['type_map'])
        delta_per_step.append(delta_c)

    all_cache_nodes = sorted(all_cache_nodes)
    node_idx = {n: i for i, n in enumerate(all_cache_nodes)}
    C = len(all_cache_nodes)
    if C == 0:
        return {}, 0.0

    # Popularity
    lam = np.zeros(F)
    for fid, sc in popularity_scores.items():
        if 0 <= fid < F:
            lam[fid] = sc
    if lam.sum() == 0:
        lam[:] = 1.0

    # Variables: x[c,f,τ] + m[c,f,τ_m]
    T = H + 1
    n_x = C * F * T
    n_m = C * F * H
    n_vars = n_x + n_m

    def x_idx(c, f, tau):
        return c * F * T + f * T + tau

    def m_idx(c, f, tau_m):
        return n_x + c * F * H + f * H + tau_m

    # Active set: which (c_idx, tau) pairs are valid
    active_set = set()
    for tau, nodes in enumerate(cache_nodes_per_step):
        for n in nodes:
            active_set.add((node_idx[n], tau))

    # Objective
    obj = np.zeros(n_vars)
    for tau in range(T):
        w = gamma ** tau
        dc = delta_per_step[tau]
        for c_name in cache_nodes_per_step[tau]:
            ci = node_idx[c_name]
            d = dc.get(c_name, 0.0)
            if d > 0:
                for f in range(F):
                    obj[x_idx(ci, f, tau)] = -w * lam[f] * d

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for c in range(C):
        for tau in range(T):
            if (c, tau) not in active_set:
                for f in range(F):
                    ub[x_idx(c, f, tau)] = 0.0

    # Build sparse A_ub
    # Count rows: capacity(C*T) + no-dup(F*T) + mig_ind(C*F*H) + mig_bud(C*H) + mig0(C)
    n_cap = C * T
    n_dup = F * T
    n_mind = C * F * H
    n_mbud = C * H
    n_m0 = C
    n_rows = n_cap + n_dup + n_mind + n_mbud + n_m0

    A = lil_matrix((n_rows, n_vars), dtype=np.float64)
    b = np.zeros(n_rows)
    row = 0

    # 1) Capacity: Σ_f x[c,f,τ] ≤ C_cap
    for c in range(C):
        for tau in range(T):
            for f in range(F):
                A[row, x_idx(c, f, tau)] = 1.0
            b[row] = C_cap
            row += 1

    # 2) Controlled redundancy: only allow a second copy when the aggregate
    # capacity can actually accommodate two full catalog replicas.
    R_MAX = 2 if (C * C_cap) >= (2 * F) else 1
    for f in range(F):
        for tau in range(T):
            for c in range(C):
                A[row, x_idx(c, f, tau)] = 1.0
            b[row] = R_MAX
            row += 1

    # Current state
    x_prev = np.zeros((C, F))
    for node_name, contents in current_cache_state.items():
        if node_name in node_idx:
            ci = node_idx[node_name]
            for fid in contents:
                if 0 <= fid < F:
                    x_prev[ci, fid] = 1.0

    # 3) Migration: -m + x(τ) - x(τ-1) ≤ rhs
    for tau_m in range(H):
        actual_tau = tau_m + 1
        for c in range(C):
            for f in range(F):
                A[row, m_idx(c, f, tau_m)] = -1.0
                A[row, x_idx(c, f, actual_tau)] = 1.0
                if actual_tau == 1:
                    b[row] = x_prev[c, f]
                else:
                    A[row, x_idx(c, f, actual_tau - 1)] = -1.0
                    b[row] = 0.0
                row += 1

    # 4) Migration budget: Σ_f m[c,f,τ_m] ≤ M_bud
    for tau_m in range(H):
        for c in range(C):
            for f in range(F):
                A[row, m_idx(c, f, tau_m)] = 1.0
            b[row] = M_bud
            row += 1

    # 5) τ=0 migration from current state
    for c in range(C):
        for f in range(F):
            if x_prev[c, f] == 0:
                A[row, x_idx(c, f, 0)] = 1.0
        b[row] = M_bud
        row += 1

    A_csc = csc_matrix(A[:row])
    b_ub = b[:row]

    # Solve
    bounds_list = list(zip(lb, ub))
    result = linprog(obj, A_ub=A_csc, b_ub=b_ub, bounds=bounds_list, method='highs')

    if not result.success:
        return current_cache_state.copy(), 0.0

    # Round τ=0: preserve the full-horizon LP signal instead of falling back
    # to popularity-only filling, which can wash out lookahead benefits.
    sol = result.x

    future_priority = np.zeros((C, F))
    start_support = np.zeros((C, F))
    for c in range(C):
        for f in range(F):
            start_support[c, f] = sol[x_idx(c, f, 0)]
            total_priority = 0.0
            for tau in range(T):
                if (c, tau) not in active_set:
                    continue
                coeff = -obj[x_idx(c, f, tau)]
                if coeff <= 0.0:
                    continue
                total_priority += coeff * sol[x_idx(c, f, tau)]
            future_priority[c, f] = total_priority

    # Collect ALL LP-recommended (value, node, content) triples
    triples = []
    for c in range(C):
        for f in range(F):
            val = start_support[c, f]
            if val > 0.01:
                triples.append((future_priority[c, f] * (1.0 + val), c, f))
    triples.sort(reverse=True)

    # Greedy assignment respecting R_MAX + capacity
    placed_count = {}
    node_count = [0] * C
    node_sets = [set() for _ in range(C)]

    for _, c, f in triples:
        if placed_count.get(f, 0) >= R_MAX:
            continue
        if node_count[c] >= C_cap:
            continue
        if f in node_sets[c]:
            continue
        node_sets[c].add(f)
        node_count[c] += 1
        placed_count[f] = placed_count.get(f, 0) + 1

    # Fill remaining capacity with the highest remaining lookahead value.
    for c in range(C):
        if node_count[c] >= C_cap:
            continue
        remaining = [
            (future_priority[c, f], start_support[c, f], lam[f], f)
            for f in range(F)
            if f not in node_sets[c] and placed_count.get(f, 0) < R_MAX
        ]
        remaining.sort(reverse=True)
        for _, _, _, f in remaining[:C_cap - node_count[c]]:
            node_sets[c].add(f)
            node_count[c] += 1
            placed_count[f] = placed_count.get(f, 0) + 1

    placement = {}
    for c in range(C):
        if node_sets[c]:
            placement[all_cache_nodes[c]] = node_sets[c]

    return placement, -result.fun

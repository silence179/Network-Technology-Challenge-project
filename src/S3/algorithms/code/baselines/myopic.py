"""
Baseline: Myopic-Optimal — same LP formulation as OLCP but with H=0 (no lookahead).
This isolates the value of the planning horizon.
"""

from ..olcp.solver import solve_olcp
from ..olcp.router import route_olcp


def solve_myopic(current_snapshot, current_cache_state, popularity_scores):
    """Solve OLCP with horizon H=0 (single step, no future topology considered)."""
    return solve_olcp([current_snapshot], current_cache_state, popularity_scores, horizon=0)


def route_myopic(G, requester, cache_nodes, placement, content_id, type_map):
    """Route using myopic-optimal placement."""
    return route_olcp(G, requester, cache_nodes, placement, content_id, type_map)

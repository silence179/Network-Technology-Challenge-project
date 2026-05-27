"""
SimPy-based discrete-event flow-level network simulator for LEO satellite caching.

Replaces the mathematical delay model (delay/(1-p)) with genuine stochastic
simulation: per-hop ARQ retransmission, queuing on bandwidth-limited links,
and real packet loss.

Each content transfer is modelled as a *flow*: it occupies each link for
the full transmission time, experiencing queuing behind other flows and
random retransmissions with exponential back-off.
"""

import simpy
import random as _random
import math
import numpy as np
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

# A content request packet is negligible compared to content itself
REQUEST_SIZE_BYTES = 1024  # 1 KB


# ── Flow result data ──

@dataclass
class FlowResult:
    """Outcome of a single content transfer through the simulated network."""
    request_id: int = 0
    src: str = ''              # the UAV requester
    dst: str = ''              # the serving node (cache SAT or GS)
    content_id: int = 0
    path: list = field(default_factory=list)
    hops: int = 0
    start_time: float = 0.0   # sim seconds
    end_time: float = 0.0
    delay_ms: float = 0.0
    retransmissions: int = 0
    delivered: bool = True
    cache_hit: bool = False
    queuing_delay_ms: float = 0.0


# ── Link model ──

class SimLink:
    """A network link modelled as a SimPy Resource with bandwidth, delay, loss."""

    def __init__(self, env: simpy.Environment, bandwidth_mbps: float,
                 prop_delay_ms: float, loss_rate: float):
        self.env = env
        self.bandwidth_bps = bandwidth_mbps * 1e6
        self.prop_delay_s = prop_delay_ms / 1000.0
        self.loss_rate = loss_rate
        # Capacity=1 serialises flows on the same link (store-and-forward)
        self.resource = simpy.Resource(env, capacity=1)
        # Statistics
        self.total_bytes = 0
        self.total_retransmissions = 0
        self.total_flows = 0

    def transfer(self, content_size_bytes: int, rng: _random.Random,
                 max_retries: int = 10):
        """SimPy generator: transfer content over this hop with stop-and-wait ARQ.

        Yields until the transfer completes or is dropped after max_retries.
        Returns (success: bool, retransmissions: int, queuing_delay_s: float).
        """
        tx_delay_s = content_size_bytes * 8 / self.bandwidth_bps

        # Acquire the link (queuing happens here)
        req = self.resource.request()
        t_queue_start = self.env.now
        yield req
        queuing_s = self.env.now - t_queue_start

        retransmissions = 0
        success = False

        for attempt in range(max_retries + 1):
            # Transmit (serialisation delay) + propagation
            yield self.env.timeout(tx_delay_s + self.prop_delay_s)

            if rng.random() >= self.loss_rate:
                # ACK received (propagation delay for ACK)
                yield self.env.timeout(self.prop_delay_s)
                success = True
                break
            else:
                # Packet lost: wait for RTO then retransmit
                retransmissions += 1
                self.total_retransmissions += 1
                # RTO ≈ 2×RTT (simplified exponential back-off capped at 4×)
                rto = min(2 * self.prop_delay_s * (2 ** min(retransmissions - 1, 2)),
                          4 * self.prop_delay_s)
                yield self.env.timeout(rto)

        self.resource.release(req)

        if success:
            self.total_bytes += content_size_bytes
            self.total_flows += 1

        return success, retransmissions, queuing_s


# ── Network simulator ──

class NetworkSimulator:
    """SimPy-based flow-level discrete event simulator.

    Usage:
        ns = NetworkSimulator(seed=42)
        ns.build_from_graph(G)               # build links from NX graph
        ns.submit_flow(path, cid, ...)       # schedule a content transfer
        ns.run(duration_s)                    # advance simulation clock
        results = ns.collect_results()        # get FlowResult list
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = _random.Random(seed)
        self.env = simpy.Environment()
        self.links: Dict[Tuple[str, str], SimLink] = {}
        self._flow_results: List[FlowResult] = []
        self._flow_counter = 0

    def build_from_graph(self, G: nx.Graph):
        """Create SimLink objects from every edge in the NetworkX topology."""
        self.links.clear()
        for u, v, data in G.edges(data=True):
            bw = data.get('bw', 100.0)       # Mbps
            delay = data.get('delay', 10.0)   # ms
            loss = data.get('loss', 0.01)
            self.links[(u, v)] = SimLink(self.env, bw, delay, loss)
            self.links[(v, u)] = SimLink(self.env, bw, delay, loss)

    def rebuild_links(self, G: nx.Graph):
        """Rebuild links for a new topology (new time step).

        Creates fresh SimPy Resources so queuing state doesn't carry over.
        """
        self.links.clear()
        for u, v, data in G.edges(data=True):
            bw = data.get('bw', 100.0)
            delay = data.get('delay', 10.0)
            loss = data.get('loss', 0.01)
            self.links[(u, v)] = SimLink(self.env, bw, delay, loss)
            self.links[(v, u)] = SimLink(self.env, bw, delay, loss)

    def _flow_process(self, path: list, content_id: int,
                      content_size_bytes: int, cache_hit: bool,
                      flow_id: int):
        """SimPy process: transfer a content flow along a multi-hop path."""
        result = FlowResult(
            request_id=flow_id,
            src=path[0],
            dst=path[-1],
            content_id=content_id,
            path=list(path),
            hops=len(path) - 1,
            cache_hit=cache_hit,
            start_time=self.env.now,
        )

        total_retransmissions = 0
        total_queuing_s = 0.0

        if cache_hit:
            # Cache hit: content delivered from cache → requester (one-way)
            # Path goes requester → cache; content flows back over same links
            for i in range(len(path) - 1, 0, -1):
                link = self.links.get((path[i], path[i - 1]))
                if link is None:
                    result.delivered = False
                    break
                success, retrans, q_delay = yield self.env.process(
                    link.transfer(content_size_bytes, self.rng)
                )
                total_retransmissions += retrans
                total_queuing_s += q_delay
                if not success:
                    result.delivered = False
                    break
        else:
            # Cache miss: request routed to origin (tiny), content returned (full)
            # Forward: small request packet, hop-by-hop
            for i in range(len(path) - 1):
                link = self.links.get((path[i], path[i + 1]))
                if link is None:
                    result.delivered = False
                    break
                success, retrans, q_delay = yield self.env.process(
                    link.transfer(REQUEST_SIZE_BYTES, self.rng)
                )
                total_retransmissions += retrans
                total_queuing_s += q_delay
                if not success:
                    result.delivered = False
                    break

            # Return: full content delivery, origin → requester
            if result.delivered:
                for i in range(len(path) - 1, 0, -1):
                    link = self.links.get((path[i], path[i - 1]))
                    if link is None:
                        result.delivered = False
                        break
                    success, retrans, q_delay = yield self.env.process(
                        link.transfer(content_size_bytes, self.rng)
                    )
                    total_retransmissions += retrans
                    total_queuing_s += q_delay
                    if not success:
                        result.delivered = False
                        break

        result.end_time = self.env.now
        result.delay_ms = (result.end_time - result.start_time) * 1000.0
        result.retransmissions = total_retransmissions
        result.queuing_delay_ms = total_queuing_s * 1000.0
        self._flow_results.append(result)

    def submit_flow(self, path: list, content_id: int,
                    content_size_bytes: int, cache_hit: bool):
        """Schedule a content transfer flow in the simulator."""
        self._flow_counter += 1
        self.env.process(
            self._flow_process(path, content_id, content_size_bytes,
                               cache_hit, self._flow_counter)
        )

    def run(self, duration_s: float = 6.0):
        """Advance the simulator clock by duration_s seconds."""
        self.env.run(until=self.env.now + duration_s)

    def run_until_done(self):
        """Run until all scheduled flows complete."""
        self.env.run()

    def collect_results(self) -> List[FlowResult]:
        """Return and clear accumulated flow results."""
        res = list(self._flow_results)
        self._flow_results.clear()
        return res

    def get_aggregate_metrics(self, flow_results: Optional[List[FlowResult]] = None):
        """Compute aggregate statistics from flow results."""
        results = flow_results if flow_results is not None else self._flow_results
        if not results:
            return {}

        delivered = [r for r in results if r.delivered]
        total = len(results)
        delays = [r.delay_ms for r in delivered]
        hit_count = sum(1 for r in delivered if r.cache_hit)
        retrans = [r.retransmissions for r in results]

        return {
            'total_requests': total,
            'delivered': len(delivered),
            'dropped': total - len(delivered),
            'delivery_rate': len(delivered) / total if total > 0 else 0,
            'avg_delay_ms': float(np.mean(delays)) if delays else 0,
            'median_delay_ms': float(np.median(delays)) if delays else 0,
            'p95_delay_ms': float(np.percentile(delays, 95)) if len(delays) >= 2 else (delays[0] if delays else 0),
            'p99_delay_ms': float(np.percentile(delays, 99)) if len(delays) >= 2 else (delays[0] if delays else 0),
            'std_delay_ms': float(np.std(delays)) if delays else 0,
            'min_delay_ms': float(min(delays)) if delays else 0,
            'max_delay_ms': float(max(delays)) if delays else 0,
            'hit_rate': hit_count / total if total > 0 else 0,
            'cache_hit_count': hit_count,
            'avg_retransmissions': float(np.mean(retrans)) if retrans else 0,
            'total_retransmissions': int(sum(retrans)),
        }

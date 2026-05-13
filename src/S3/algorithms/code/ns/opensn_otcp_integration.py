"""Export OTCP and baseline snapshots as OpenSN artifacts and optionally replay them.

Usage:
    python -m code.ns.opensn_otcp_integration --method otcp --max-steps 20
    python -m code.ns.opensn_otcp_integration --method all --max-steps 5 --apply
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from ipaddress import IPv4Address
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from .. import config as cfg
from ..config import GREEDY_REFRESH_INTERVAL
from ..common.metrics import generate_requests
from ..common.popularity import PopularityTracker
from ..common.topology import get_all_reachable_sats, get_cache_nodes
from ..olcp.solver import solve_olcp
from ..baselines.drl_actor_critic import MaDRLManager, route_madrl
from ..baselines.greedy_popular import greedy_placement
from ..baselines.lce_lru import LCELRUManager
from ..baselines.myopic import solve_myopic
from ..baselines.spacecache_plus import spacecache_placement
from ..baselines.submodular_greedy import submodular_greedy_placement
from ..repro.common import StepSnapshot, build_snapshot, load_sampled_traces
from .opensn_metrics import timed_request, wait_for_activation, wait_for_platform
from .opensn_otcp_controller import push_configs, request_json


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / 'results'
DEFAULT_BASE_URL = 'http://127.0.0.1:8080'
DEFAULT_IMAGE = 'opensn/local-node:latest'
DEFAULT_RUNTIME_STATUS_DIR = Path(__file__).resolve().parents[3] / 'OpenSN-Library' / 'daemon' / 'runtime' / 'share' / 'otcp_status'
SUPPORTED_METHODS = ('nocache', 'lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'otcp')
METHOD_ALIASES = {'olcp': 'otcp'}


def _normalize_method(method: str) -> str:
    normalized = METHOD_ALIASES.get(method, method)
    if normalized not in SUPPORTED_METHODS and normalized != 'all':
        raise ValueError(f'Unsupported method: {method}')
    return normalized


def _opensn_type(node_type: str) -> str:
    if node_type == 'SAT':
        return 'Satellite'
    if node_type == 'GS':
        return 'GroundStation'
    return 'GroundTerminal'


def _stringify_map(payload: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in payload.items()}


def _count_request_failures(service_results: Counter | dict[str, int]) -> int:
    failures = 0
    for result, count in service_results.items():
        result_name = str(result)
        if result_name.endswith('_failed') or result_name in {'miss', 'unknown'}:
            failures += int(count)
    return failures


@contextmanager
def _runtime_overrides(args):
    previous = {
        'CACHE_CAPACITY': cfg.CACHE_CAPACITY,
        'LCE_LRU_CAPACITY': cfg.LCE_LRU_CAPACITY,
        'ZIPF_ALPHA': cfg.ZIPF_ALPHA,
        'REQUESTS_PER_STEP': cfg.REQUESTS_PER_STEP,
        'MIGRATION_BUDGET': cfg.MIGRATION_BUDGET,
    }
    if args.cache_capacity is not None:
        cfg.CACHE_CAPACITY = args.cache_capacity
        cfg.LCE_LRU_CAPACITY = args.cache_capacity
    if args.zipf_alpha is not None:
        cfg.ZIPF_ALPHA = args.zipf_alpha
    if args.requests_per_step is not None:
        cfg.REQUESTS_PER_STEP = args.requests_per_step
    if args.migration_budget is not None:
        cfg.MIGRATION_BUDGET = args.migration_budget
    try:
        yield
    finally:
        cfg.CACHE_CAPACITY = previous['CACHE_CAPACITY']
        cfg.LCE_LRU_CAPACITY = previous['LCE_LRU_CAPACITY']
        cfg.ZIPF_ALPHA = previous['ZIPF_ALPHA']
        cfg.REQUESTS_PER_STEP = previous['REQUESTS_PER_STEP']
        cfg.MIGRATION_BUDGET = previous['MIGRATION_BUDGET']


def _copy_placement(placement: dict[str, set[int]] | dict[str, list[int]]) -> dict[str, set[int]]:
    copied: dict[str, set[int]] = {}
    for node_id, contents in placement.items():
        copied[node_id] = {int(item) for item in contents}
    return copied


def _placement_manifest(placement: dict[str, set[int]]) -> dict[str, list[int]]:
    return {node: sorted(int(item) for item in contents) for node, contents in placement.items()}


def _best_serving_node(G, requester: str, placement: dict[str, set[int]], content_id: int):
    if not G.has_node(requester):
        return None

    import networkx as nx

    best_delay = float('inf')
    best_node = None
    for node, cached_items in placement.items():
        if content_id not in cached_items or not G.has_node(node):
            continue
        try:
            path = nx.dijkstra_path(G, requester, node, weight='eff_delay')
            delay = sum(G[path[index]][path[index + 1]]['eff_delay'] for index in range(len(path) - 1))
            if delay < best_delay:
                best_delay = delay
                best_node = node
        except nx.NetworkXNoPath:
            continue
    return best_node


def _build_future_snapshots(df_sat, df_uav, timestamps, start_idx: int, horizon: int):
    snapshots = []
    for offset in range(start_idx, min(start_idx + horizon + 1, len(timestamps))):
        snapshot = build_snapshot(df_sat, df_uav, timestamps[offset])
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def _trim_snapshot(snapshot: StepSnapshot, max_satellites: int | None, max_uavs: int | None) -> StepSnapshot | None:
    if max_satellites is None and max_uavs is None:
        return snapshot

    sat_nodes = sorted(node for node, node_type in snapshot.type_map.items() if node_type == 'SAT')
    uav_nodes = sorted(node for node, node_type in snapshot.type_map.items() if node_type == 'UAV')
    gs_nodes = sorted(node for node, node_type in snapshot.type_map.items() if node_type == 'GS')

    kept_uavs = uav_nodes[:max_uavs] if max_uavs is not None else uav_nodes

    if max_satellites is None:
        kept_sats = sat_nodes
    else:
        priority_sats = []
        for node in sorted(snapshot.all_reachable | snapshot.cache_nodes):
            if node in sat_nodes and node not in priority_sats:
                priority_sats.append(node)
        remaining_sats = [node for node in sat_nodes if node not in priority_sats]
        kept_sats = (priority_sats + remaining_sats)[:max_satellites]

    selected_nodes = set(gs_nodes) | set(kept_uavs) | set(kept_sats)
    if len(selected_nodes) < 2:
        return None

    trimmed_graph = snapshot.G.subgraph(selected_nodes).copy()
    trimmed_type_map = {node: snapshot.type_map[node] for node in trimmed_graph.nodes}
    trimmed_coord_map = {node: snapshot.coord_map[node] for node in trimmed_graph.nodes if node in snapshot.coord_map}
    trimmed_cache_nodes = get_cache_nodes(trimmed_graph, trimmed_type_map)
    trimmed_all_reachable = get_all_reachable_sats(trimmed_graph, trimmed_type_map)
    if not trimmed_all_reachable:
        return None

    return StepSnapshot(
        t_ms=snapshot.t_ms,
        G=trimmed_graph,
        type_map=trimmed_type_map,
        coord_map=trimmed_coord_map,
        cache_nodes=set(trimmed_cache_nodes),
        all_reachable=set(trimmed_all_reachable),
    )


def _make_emulation_config(args) -> dict[str, Any]:
    resource_limit = {
        'nano_cpu': args.nano_cpu,
        'memory_byte': args.memory_byte,
    }
    return {
        'Satellite': {
            'image': args.satellite_image,
            'container_envs': {},
            'resource_limit': resource_limit,
        },
        'GroundStation': {
            'image': args.ground_image,
            'container_envs': {},
            'resource_limit': resource_limit,
        },
        'GroundTerminal': {
            'image': args.terminal_image,
            'container_envs': {},
            'resource_limit': resource_limit,
        },
    }


def _allocate_link_addresses(link_index: int) -> list[dict[str, str]]:
    network_base = int(IPv4Address('10.200.0.0')) + (link_index * 4)
    return [
        {'IPV4': f'{IPv4Address(network_base + 1)}/30'},
        {'IPV4': f'{IPv4Address(network_base + 2)}/30'},
    ]


def _build_topology_payload(snapshot, placement: dict[str, set[int]], step_index: int, method: str) -> dict[str, Any]:
    nodes = sorted(snapshot.G.nodes)
    node_to_index = {node_id: index for index, node_id in enumerate(nodes)}

    instances = []
    for node_id in nodes:
        node_type = snapshot.type_map[node_id]
        coords = snapshot.coord_map.get(node_id, (0.0, 0.0, 0.0))
        cached = sorted(int(item) for item in placement.get(node_id, set()))
        extra = {
            'OTCPNodeId': node_id,
            'OTCPOriginalType': node_type,
            'OTCPStep': step_index,
            'OTCPMethod': method,
            'OTCPCachedContents': ','.join(str(item) for item in cached),
            'OTCPCacheCount': len(cached),
            'ecef_x': coords[0],
            'ecef_y': coords[1],
            'ecef_z': coords[2],
        }
        instances.append({
            'type': _opensn_type(node_type),
            'extra': _stringify_map(extra),
            'device_info': {},
        })

    links = []
    for link_index, (node_a, node_b, edge) in enumerate(sorted(snapshot.G.edges(data=True))):
        links.append({
            'end_indexes': [node_to_index[node_a], node_to_index[node_b]],
            'type': 'vlink',
            'init_parameter': {
                'connect': 1,
                'delay': int(round(float(edge.get('delay', 0.0)) * 1000.0)),
                'bandwidth': int(round(float(edge.get('bw', 0.0)) * 1_000_000.0)),
                'loss': int(round(float(edge.get('loss', 0.0)) * 10000.0)),
            },
            'address_infos': _allocate_link_addresses(link_index),
            'extra': _stringify_map({
                'eff_delay_ms': edge.get('eff_delay', 0.0),
                'dist_m': edge.get('dist_m', 0.0),
            }),
        })

    return {
        'instances': instances,
        'links': links,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


@contextmanager
def _temporary_request_seed(seed: int | None):
    if seed is None:
        yield
        return

    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def _read_runtime_payloads(status_dir: Path, live_instance_ids: set[str], step: int) -> list[dict[str, Any]]:
    try:
        exists = status_dir.exists()
    except PermissionError:
        return []
    if not exists:
        return []

    payloads = []
    try:
        status_files = sorted(status_dir.glob('*.json'))
    except PermissionError:
        return []

    for status_file in status_files:
        try:
            payload = json.loads(status_file.read_text())
        except (OSError, PermissionError, json.JSONDecodeError):
            continue
        if payload.get('instance_id') not in live_instance_ids:
            continue
        if int(payload.get('step', -1)) != step:
            continue
        payloads.append(payload)
    return payloads


def _initial_runtime_state(method: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        'tracker': PopularityTracker(),
        'placement': {},
    }
    if method == 'lce_lru':
        state['lce_mgr'] = LCELRUManager()
    if method == 'madrl':
        state['madrl_mgr'] = MaDRLManager()
    return state


def _snapshot_view(snapshot: StepSnapshot) -> dict[str, Any]:
    return {
        'G': snapshot.G,
        'cache_nodes': snapshot.all_reachable,
        'type_map': snapshot.type_map,
    }


def _cold_start_zipf_prior() -> dict[int, float]:
    ranks = np.arange(1, cfg.CONTENT_CATALOG_SIZE + 1, dtype=float)
    weights = np.power(ranks, -cfg.ZIPF_ALPHA)
    weights /= weights.sum()
    return {int(content_id): float(weight) for content_id, weight in enumerate(weights)}


def _request_batch_scores(request_batch: list[tuple[str, int]]) -> dict[int, float]:
    demand = Counter()
    for _, content_id in request_batch:
        demand[int(content_id)] += 1
    return {content_id: float(score) for content_id, score in demand.items()}


def _nearest_cache_node(snapshot: StepSnapshot, requester: str) -> str | None:
    if not snapshot.G.has_node(requester):
        return None

    import networkx as nx

    best_node = None
    best_cost = float('inf')
    for node_id in sorted(snapshot.all_reachable):
        if not snapshot.G.has_node(node_id):
            continue
        try:
            cost = float(nx.dijkstra_path_length(snapshot.G, requester, node_id, weight='eff_delay'))
        except nx.NetworkXNoPath:
            continue
        if cost < best_cost or (cost == best_cost and (best_node is None or node_id < best_node)):
            best_node = node_id
            best_cost = cost
    return best_node


def _assign_requested_content_locally(
    snapshot: StepSnapshot,
    placement: dict[str, set[int]],
    request_batch: list[tuple[str, int]],
) -> dict[str, set[int]]:
    if not request_batch:
        return placement

    request_scores = _request_batch_scores(request_batch)
    requesters_by_content: dict[int, list[str]] = {}
    for requester, content_id in request_batch:
        requesters_by_content.setdefault(int(content_id), []).append(requester)

    adjusted = _copy_placement(placement)
    for content_id, requesters in sorted(
        requesters_by_content.items(),
        key=lambda item: (-request_scores.get(item[0], 0.0), item[0]),
    ):
        preferred_nodes = []
        seen_nodes = set()
        for requester in sorted(set(requesters)):
            node_id = _nearest_cache_node(snapshot, requester)
            if node_id is None or node_id in seen_nodes:
                continue
            preferred_nodes.append(node_id)
            seen_nodes.add(node_id)

        for node_id in preferred_nodes:
            contents = adjusted.setdefault(node_id, set())
            if content_id in contents:
                continue
            if len(contents) >= cfg.CACHE_CAPACITY:
                victim = min(
                    contents,
                    key=lambda existing: (
                        1 if request_scores.get(existing, 0.0) > 0.0 else 0,
                        request_scores.get(existing, 0.0),
                        existing,
                    ),
                )
                contents.remove(victim)
            contents.add(content_id)

    return adjusted


def _solve_method_placement(
    method: str,
    step_index: int,
    future_snapshots: list[StepSnapshot],
    runtime_state: dict[str, Any],
    request_batch: list[tuple[str, int]] | None = None,
) -> tuple[dict[str, set[int]], dict[str, float]]:
    current = future_snapshots[0]
    tracker = runtime_state['tracker']
    pop_scores = dict(tracker.scores)
    placement = runtime_state['placement']
    solve_metrics: dict[str, float] = {'solve_time_ms': 0.0}

    start = time.perf_counter()
    if method == 'otcp':
        if request_batch:
            for content_id, score in _request_batch_scores(request_batch).items():
                pop_scores[content_id] = pop_scores.get(content_id, 0.0) + score
        if not pop_scores:
            pop_scores = _cold_start_zipf_prior()
        placement, _ = solve_olcp([_snapshot_view(snapshot) for snapshot in future_snapshots], placement, pop_scores)
        placement = _assign_requested_content_locally(current, placement, request_batch or [])
    elif method == 'myopic':
        placement, _ = solve_myopic(_snapshot_view(current), placement, pop_scores)
    elif method == 'greedy':
        if step_index % GREEDY_REFRESH_INTERVAL == 0 or not placement:
            placement = greedy_placement(current.cache_nodes, tracker)
    elif method == 'submod':
        if step_index % GREEDY_REFRESH_INTERVAL == 0 or not placement:
            placement = submodular_greedy_placement(current.G, current.all_reachable, current.type_map, pop_scores)
    elif method == 'spacecache':
        if step_index % GREEDY_REFRESH_INTERVAL == 0 or not placement:
            placement = spacecache_placement(current.G, current.all_reachable, current.type_map, pop_scores)
    elif method == 'madrl':
        placement = runtime_state['madrl_mgr'].decide_placement(current.all_reachable, pop_scores)
    elif method == 'lce_lru':
        placement = {
            node_id: store.contents()
            for node_id, store in runtime_state['lce_mgr']._stores.items()
        }
    elif method == 'nocache':
        placement = {}
    else:
        raise RuntimeError(f'Unhandled method: {method}')
    solve_metrics['solve_time_ms'] = (time.perf_counter() - start) * 1000.0

    copied = _copy_placement(placement)
    runtime_state['placement'] = copied
    return copied, {'solve_time_ms': round(solve_metrics['solve_time_ms'], 3), 'popularity_keys': len(pop_scores)}


def _advance_method_state(
    method: str,
    current: StepSnapshot,
    request_batch: list[tuple[str, int]],
    placement: dict[str, set[int]],
    runtime_state: dict[str, Any],
    popularity_scores: dict[str, float],
) -> None:
    tracker = runtime_state['tracker']

    if method == 'lce_lru':
        cache_nodes = sorted(current.cache_nodes)
        for requester, content_id in request_batch:
            runtime_state['lce_mgr'].route(current.G, requester, cache_nodes, content_id, current.type_map)
        runtime_state['placement'] = {
            node_id: store.contents()
            for node_id, store in runtime_state['lce_mgr']._stores.items()
        }
    elif method == 'madrl':
        hit_counts = {node_id: 0 for node_id in current.all_reachable}
        for requester, content_id in request_batch:
            serving_node = _best_serving_node(current.G, requester, placement, content_id)
            _, _, hit = route_madrl(current.G, requester, placement, content_id, current.type_map)
            if hit and serving_node is not None:
                hit_counts[serving_node] = hit_counts.get(serving_node, 0) + 1
        runtime_state['madrl_mgr'].feedback(current.all_reachable, hit_counts, popularity_scores)

    tracker.decay_all()
    for _, content_id in request_batch:
        tracker.record(content_id)


def export_sequence(args, method: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'seed', None) is not None:
        method_seed = int(args.seed) + (SUPPORTED_METHODS.index(method) * 1000)
        random.seed(method_seed)
        np.random.seed(method_seed % (2**32))

    df_sat, df_uav, timestamps = load_sampled_traces(sat_dir=str(args.sat_dir), max_steps=args.max_steps)
    if not timestamps:
        raise RuntimeError('No sampled timestamps available for OpenSN OTCP integration.')

    emu_config = _make_emulation_config(args)
    _write_json(output_dir / 'emu_config.json', emu_config)

    runtime_state = _initial_runtime_state(method)
    exported_steps = []

    for step_index in range(len(timestamps)):
        future_snapshots = []
        for snapshot in _build_future_snapshots(df_sat, df_uav, timestamps, step_index, cfg.LOOKAHEAD_HORIZON):
            trimmed = _trim_snapshot(snapshot, args.max_satellites, args.max_uavs)
            if trimmed is not None:
                future_snapshots.append(trimmed)
        if not future_snapshots:
            continue

        current = future_snapshots[0]
        request_seed = None
        if getattr(args, 'seed', None) is not None:
            request_seed = int(args.seed) + step_index
        with _temporary_request_seed(request_seed):
            request_batch = generate_requests(current.G, current.type_map, n_req=args.requests_per_step)

        popularity_scores = dict(runtime_state['tracker'].scores)
        placement, solve_metrics = _solve_method_placement(
            method,
            step_index,
            future_snapshots,
            runtime_state,
            request_batch=request_batch,
        )

        topology_payload = _build_topology_payload(current, placement, step_index, method)
        topology_path = output_dir / f'topology_step_{step_index:03d}.json'
        _write_json(topology_path, topology_payload)

        request_payload = {
            'step': step_index,
            'timestamp_ms': current.t_ms,
            'method': method,
            'requests': [
                {'requester': requester, 'content_id': int(content_id)}
                for requester, content_id in request_batch
            ],
        }
        request_path = output_dir / f'requests_step_{step_index:03d}.json'
        _write_json(request_path, request_payload)

        placement_payload = {
            'step': step_index,
            'timestamp_ms': current.t_ms,
            'method': method,
            'placement': _placement_manifest(placement),
        }
        placement_path = output_dir / f'placement_step_{step_index:03d}.json'
        _write_json(placement_path, placement_payload)

        _advance_method_state(method, current, request_batch, placement, runtime_state, popularity_scores)

        exported_steps.append({
            'step': step_index,
            'timestamp_ms': current.t_ms,
            'instance_count': len(topology_payload['instances']),
            'link_count': len(topology_payload['links']),
            'request_count': len(request_batch),
            'solve_time_ms': solve_metrics['solve_time_ms'],
            'topology_file': topology_path.name,
            'request_file': request_path.name,
            'placement_file': placement_path.name,
        })

    manifest = {
        'platform': 'OpenSN',
        'source': 'OTCP and baseline dynamic trace export',
        'method': method,
        'lookahead_horizon': cfg.LOOKAHEAD_HORIZON,
        'step_stride': cfg.STEP_STRIDE,
        'sat_dir': str(args.sat_dir),
        'max_satellites': args.max_satellites,
        'max_uavs': args.max_uavs,
        'requests_per_step': args.requests_per_step or cfg.REQUESTS_PER_STEP,
        'request_seed_base': getattr(args, 'seed', None),
        'cache_capacity': cfg.CACHE_CAPACITY,
        'zipf_alpha': cfg.ZIPF_ALPHA,
        'migration_budget': cfg.MIGRATION_BUDGET,
        'sequence_dir': str(output_dir),
        'emu_config_file': 'emu_config.json',
        'steps': exported_steps,
    }
    _write_json(output_dir / 'sequence_manifest.json', manifest)
    return manifest


def _collect_runtime_status(
    base_url: str,
    output_dir: Path,
    step_entry: dict[str, Any],
    runtime_status_dir: Path | None,
) -> dict[str, Any] | None:
    summaries = request_json(base_url, '/api/instance/')['data']
    if not summaries:
        return None

    live_instance_ids = {item['instance_id'] for item in summaries}

    raw_payloads: list[dict[str, Any]] = []
    status_source = None

    if runtime_status_dir is not None:
        raw_payloads = _read_runtime_payloads(runtime_status_dir, live_instance_ids, int(step_entry['step']))
        if raw_payloads:
            status_source = str(runtime_status_dir)

    if not raw_payloads:
        container_name = summaries[0].get('name')
        if not container_name:
            return None

        with tempfile.TemporaryDirectory(prefix='opensn-status-') as temp_dir:
            destination = Path(temp_dir) / 'otcp_status'
            copy = subprocess.run(
                ['docker', 'cp', f'{container_name}:/share/otcp_status', str(destination)],
                capture_output=True,
                text=True,
                check=False,
            )
            if copy.returncode != 0:
                return {
                    'step': step_entry['step'],
                    'status': 'docker_cp_failed',
                    'stderr': copy.stderr.strip(),
                }

            raw_status_dir = destination / 'otcp_status'
            if not raw_status_dir.exists():
                raw_status_dir = destination
            raw_payloads = _read_runtime_payloads(raw_status_dir, live_instance_ids, int(step_entry['step']))
            if raw_payloads:
                status_source = f'docker:{container_name}'

    service_results = Counter()
    totals = Counter()
    for payload in raw_payloads:
        totals['local_hits'] += int(payload.get('local_hits', 0))
        totals['adjacent_hits'] += int(payload.get('adjacent_hits', 0))
        totals['global_hits'] += int(payload.get('global_hits', 0))
        totals['served_locally'] += int(payload.get('served_locally', 0))
        totals['served_from_neighbor'] += int(payload.get('served_from_neighbor', 0))
        totals['served_from_global'] += int(payload.get('served_from_global', 0))
        totals['served_from_origin'] += int(payload.get('served_from_origin', 0))
        totals['fetch_failures'] += int(payload.get('fetch_failures', 0))
        totals['pending_requests'] += int(payload.get('pending_request_count', 0))
        for evaluation in payload.get('request_evaluations') or []:
            service_results[evaluation.get('service_result', 'unknown')] += 1

    request_total = totals['pending_requests'] or sum(service_results.values())
    evaluated_requests = sum(service_results.values())
    request_failures = _count_request_failures(service_results)
    cache_hits = (
        service_results.get('local_cache', 0)
        + service_results.get('adjacent_http', 0)
        + service_results.get('global_http', 0)
    )
    origin_serves = service_results.get('origin_http', 0)

    summary = {
        'step': step_entry['step'],
        'timestamp_ms': step_entry['timestamp_ms'],
        'status_file_count': len(raw_payloads),
        'live_instance_count': len(live_instance_ids),
        'status_source': status_source,
        'local_hits': totals['local_hits'],
        'adjacent_hits': totals['adjacent_hits'],
        'global_hits': totals['global_hits'],
        'served_locally': totals['served_locally'],
        'served_from_neighbor': totals['served_from_neighbor'],
        'served_from_global': totals['served_from_global'],
        'served_from_origin': totals['served_from_origin'],
        'fetch_failures': request_failures,
        'fetch_attempt_failures': totals['fetch_failures'],
        'pending_requests': totals['pending_requests'],
        'evaluated_requests': evaluated_requests,
        'request_failures': request_failures,
        'cache_hit_rate': (cache_hits / request_total) if request_total else 0.0,
        'origin_rate': (origin_serves / request_total) if request_total else 0.0,
        'service_results': dict(sorted(service_results.items())),
    }
    return summary


def _runtime_status_is_complete(summary: dict[str, Any] | None, expected_requests: int) -> bool:
    if not summary or summary.get('status'):
        return False

    status_file_count = int(summary.get('status_file_count', 0))
    live_instance_count = int(summary.get('live_instance_count', 0))
    pending_requests = int(summary.get('pending_requests', 0))
    evaluated_requests = int(summary.get('evaluated_requests', 0))
    return (
        status_file_count > 0
        and live_instance_count > 0
        and status_file_count >= live_instance_count
        and pending_requests >= expected_requests
        and evaluated_requests >= expected_requests
    )


def _wait_for_runtime_status(
    base_url: str,
    output_dir: Path,
    step_entry: dict[str, Any],
    timeout: float,
    runtime_status_dir: Path | None,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout
    last_summary = None
    best_summary = None
    expected_requests = int(step_entry.get('request_count', 0))
    while time.time() <= deadline:
        try:
            last_summary = _collect_runtime_status(base_url, output_dir, step_entry, runtime_status_dir)
        except Exception as exc:
            last_summary = {
                'step': step_entry['step'],
                'status': 'runtime_status_request_error',
                'error': str(exc),
            }
            time.sleep(2.0)
            continue
        if last_summary is not None:
            if best_summary is None:
                best_summary = last_summary
            else:
                current_files = int(last_summary.get('status_file_count', 0))
                best_files = int(best_summary.get('status_file_count', 0))
                current_requests = int(last_summary.get('pending_requests', 0))
                best_requests = int(best_summary.get('pending_requests', 0))
                current_evaluated = int(last_summary.get('evaluated_requests', 0))
                best_evaluated = int(best_summary.get('evaluated_requests', 0))
                if (current_files, current_requests, current_evaluated) >= (best_files, best_requests, best_evaluated):
                    best_summary = last_summary

            if _runtime_status_is_complete(last_summary, expected_requests):
                completed_summary = dict(last_summary)
                completed_summary['expected_requests'] = expected_requests
                completed_summary['runtime_complete'] = True
                _write_json(output_dir / f'runtime_step_{step_entry["step"]:03d}.json', completed_summary)
                return completed_summary
        time.sleep(2.0)

    incomplete_summary = best_summary or last_summary
    if incomplete_summary is None:
        return None

    rejected_summary = dict(incomplete_summary)
    rejected_summary['expected_requests'] = expected_requests
    rejected_summary['runtime_complete'] = False
    rejected_summary.setdefault('status', 'runtime_status_incomplete')
    _write_json(output_dir / f'runtime_step_{step_entry["step"]:03d}_incomplete.json', rejected_summary)
    return rejected_summary


def replay_sequence(args, manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    emu_config_text = (output_dir / manifest['emu_config_file']).read_text()
    manifest_path = output_dir / 'sequence_manifest.json'
    wait_for_platform(args.base_url, args.timeout)

    replay_steps = []
    started = False
    if not args.reset_each_step:
        timed_request(args.base_url, '/api/emulation/reset', method='POST', payload='')
        timed_request(args.base_url, '/api/emulation/update', method='POST', payload=emu_config_text)

    for entry in manifest['steps'][:args.apply_steps or None]:
        topology_text = (output_dir / entry['topology_file']).read_text()
        step_metrics: dict[str, Any] = {
            'step': entry['step'],
            'timestamp_ms': entry['timestamp_ms'],
            'request_count': entry.get('request_count', 0),
        }

        if args.reset_each_step or not started:
            timed_request(args.base_url, '/api/emulation/reset', method='POST', payload='')
            timed_request(args.base_url, '/api/emulation/update', method='POST', payload=emu_config_text)
            started = False

        _, step_metrics['topology_seconds'] = timed_request(
            args.base_url,
            '/api/emulation/topology',
            method='POST',
            payload=topology_text,
        )
        if not started:
            _, step_metrics['start_seconds'] = timed_request(
                args.base_url,
                '/api/emulation/start',
                method='POST',
                payload='',
            )
            started = True
        else:
            step_metrics['start_seconds'] = 0.0

        activation_snapshot, step_metrics['activation_seconds'] = wait_for_activation(
            args.base_url,
            entry['instance_count'],
            entry['link_count'],
            args.timeout,
        )
        step_metrics.update(activation_snapshot)
        step_metrics['config_push'] = push_configs(args.base_url, manifest_path, int(entry['step']), args.verify_configs)

        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)

        if args.collect_runtime_status:
            step_metrics['runtime'] = _wait_for_runtime_status(
                args.base_url,
                output_dir,
                entry,
                args.runtime_timeout,
                args.runtime_status_dir,
            )

        replay_steps.append(step_metrics)

    replay_payload = {
        'base_url': args.base_url,
        'method': manifest['method'],
        'reset_each_step': args.reset_each_step,
        'steps': replay_steps,
    }
    _write_json(output_dir / 'replay_metrics.json', replay_payload)
    return replay_payload


def run_workflow(args, method: str, output_dir: Path) -> dict[str, Any]:
    with _runtime_overrides(args):
        manifest = export_sequence(args, method, output_dir)
        payload: dict[str, Any] = {'manifest': manifest}
        if args.apply:
            payload['replay'] = replay_sequence(args, manifest, output_dir)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Export OTCP or baseline sequences to OpenSN and optionally replay them.')
    parser.add_argument('--method', default='otcp', help='One of otcp, nocache, lce_lru, greedy, madrl, submod, spacecache, myopic, all')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--sat-dir', type=Path, default=Path(cfg.SAT_DIR))
    parser.add_argument('--max-satellites', type=int, default=None)
    parser.add_argument('--max-uavs', type=int, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--satellite-image', default=DEFAULT_IMAGE)
    parser.add_argument('--ground-image', default=DEFAULT_IMAGE)
    parser.add_argument('--terminal-image', default=DEFAULT_IMAGE)
    parser.add_argument('--nano-cpu', default='50M')
    parser.add_argument('--memory-byte', default='128M')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--apply-steps', type=int, default=0)
    parser.add_argument('--reset-each-step', action='store_true')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--settle-seconds', type=float, default=6.0)
    parser.add_argument('--runtime-timeout', type=float, default=45.0)
    parser.add_argument('--runtime-status-dir', type=Path, default=DEFAULT_RUNTIME_STATUS_DIR)
    parser.add_argument('--requests-per-step', type=int, default=None)
    parser.add_argument('--cache-capacity', type=int, default=None)
    parser.add_argument('--zipf-alpha', type=float, default=None)
    parser.add_argument('--migration-budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-configs', action='store_true')
    parser.add_argument('--collect-runtime-status', action='store_true')
    args = parser.parse_args()

    method = _normalize_method(args.method)
    methods = list(SUPPORTED_METHODS) if method == 'all' else [method]

    payload: dict[str, Any] = {}
    for selected_method in methods:
        output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f'opensn_{selected_method}')
        if method == 'all' and args.output_dir is not None:
            output_dir = args.output_dir / selected_method
        payload[selected_method] = run_workflow(args, selected_method, output_dir)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
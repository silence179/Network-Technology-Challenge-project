"""Push OTCP placement and request batches into OpenSN instance configs.

Usage:
    python -m code.ns.opensn_otcp_controller \
        --manifest results/opensn_otcp_smoke/sequence_manifest.json \
        --step 0
"""

from __future__ import annotations

import argparse
import base64
import json
import networkx as nx
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = 'http://127.0.0.1:8080'
DEFAULT_HTTP_TIMEOUT = 60.0
DEFAULT_HTTP_RETRIES = 3


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = 'GET',
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    retries: int = DEFAULT_HTTP_RETRIES,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(base_url + path, method=method, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < max(retries, 1):
                time.sleep(retry_delay)
    if last_error is None:
        raise RuntimeError(f'Failed to read OpenSN response for {path}')
    raise last_error


def get_etcd_base_url(base_url: str) -> str:
    data = request_json(base_url, '/api/platform/address/etcd')
    endpoint = data['data']
    return f"http://{endpoint['address']}:{endpoint['port']}"


def put_etcd_value(etcd_base_url: str, key: str, value: str) -> None:
    payload = {
        'key': base64.b64encode(key.encode()).decode(),
        'value': base64.b64encode(value.encode()).decode(),
    }
    request_json(etcd_base_url, '/v3/kv/put', method='POST', payload=payload)


def get_etcd_value(etcd_base_url: str, key: str) -> str | None:
    payload = {'key': base64.b64encode(key.encode()).decode()}
    data = request_json(etcd_base_url, '/v3/kv/range', method='POST', payload=payload)
    kvs = data.get('kvs') or []
    if not kvs:
        return None
    return base64.b64decode(kvs[0]['value']).decode()


def load_step_artifacts(manifest_path: Path, step: int | None):
    manifest = json.loads(manifest_path.read_text())
    steps = manifest.get('steps', [])
    if not steps:
        raise RuntimeError('Manifest contains no exported steps.')

    if step is None:
        entry = steps[0]
    else:
        matches = [item for item in steps if int(item['step']) == step]
        if not matches:
            raise RuntimeError(f'Step {step} not found in manifest.')
        entry = matches[0]

    sequence_dir = manifest_path.parent
    placement_payload = json.loads((sequence_dir / entry['placement_file']).read_text())
    request_payload = json.loads((sequence_dir / entry['request_file']).read_text())
    topology_payload = json.loads((sequence_dir / entry['topology_file']).read_text())
    return manifest, entry, topology_payload, placement_payload, request_payload


def build_request_map(request_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in request_payload.get('requests', []):
        grouped.setdefault(item['requester'], []).append(item)
    return grouped


def build_instance_index(base_url: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    summaries = request_json(base_url, '/api/instance/')['data']
    summary_by_id = {item['instance_id']: item for item in summaries}
    detail_by_id: dict[str, dict[str, Any]] = {}
    for item in summaries:
        detail = request_json(base_url, f"/api/instance/{item['node_index']}/{item['instance_id']}")['data']
        detail_by_id[item['instance_id']] = detail
    return summary_by_id, detail_by_id


def build_link_detail_cache(base_url: str, detail_by_id: dict[str, dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    cache: dict[tuple[int, str], dict[str, Any]] = {}
    for detail in detail_by_id.values():
        for link_id, conn in (detail.get('connections') or {}).items():
            key = (conn['end_node_index'], link_id)
            if key in cache:
                continue
            try:
                cache[key] = request_json(base_url, f"/api/link/{detail['node_index']}/{link_id}")['data']
            except Exception:
                cache[key] = request_json(base_url, f"/api/link/{conn['end_node_index']}/{link_id}")['data']
    return cache


def _link_weight(link_detail: dict[str, Any]) -> float:
    extra = link_detail.get('extra') or {}
    try:
        return float(extra.get('eff_delay_ms', 1.0))
    except (TypeError, ValueError):
        return 1.0


def build_route_plan(
    summary_by_id: dict[str, dict[str, Any]],
    detail_by_id: dict[str, dict[str, Any]],
    link_cache: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    graph = nx.Graph()
    instance_to_node: dict[str, str] = {}
    node_types: dict[str, str] = {}

    for instance_id, summary in summary_by_id.items():
        extra = summary.get('extra') or {}
        otcp_node_id = extra.get('OTCPNodeId', instance_id)
        instance_to_node[instance_id] = otcp_node_id
        node_types[otcp_node_id] = extra.get('OTCPOriginalType', summary.get('type', ''))
        graph.add_node(otcp_node_id)

    seen_edges: set[tuple[str, str]] = set()
    for detail in detail_by_id.values():
        for link_id, conn in (detail.get('connections') or {}).items():
            link_detail = link_cache.get((detail['node_index'], link_id)) or link_cache.get((conn['end_node_index'], link_id))
            if not link_detail:
                continue

            end_infos = link_detail.get('end_infos') or []
            if len(end_infos) < 2:
                continue

            node_a = instance_to_node.get(end_infos[0].get('instance_id', ''))
            node_b = instance_to_node.get(end_infos[1].get('instance_id', ''))
            if not node_a or not node_b or node_a == node_b:
                continue

            edge_key = tuple(sorted((node_a, node_b)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            graph.add_edge(node_a, node_b, weight=_link_weight(link_detail))

    origin_nodes = {node_id for node_id, node_type in node_types.items() if node_type == 'GS'}
    route_plan: dict[str, dict[str, dict[str, Any]]] = {}
    origin_by_node: dict[str, str] = {}

    for node_id in graph.nodes:
        try:
            lengths, paths = nx.single_source_dijkstra(graph, node_id, weight='weight')
        except nx.NetworkXNoPath:
            lengths, paths = {node_id: 0.0}, {node_id: [node_id]}

        node_routes: dict[str, dict[str, Any]] = {}
        for target_id, path in paths.items():
            if target_id == node_id or len(path) < 2:
                continue
            node_routes[target_id] = {
                'next_hop_node_id': path[1],
                'hop_count': len(path) - 1,
                'path_cost': float(lengths.get(target_id, len(path) - 1)),
            }
        route_plan[node_id] = node_routes

        closest_origin = None
        closest_cost = float('inf')
        for origin_id in origin_nodes:
            if origin_id not in lengths:
                continue
            if lengths[origin_id] < closest_cost:
                closest_cost = float(lengths[origin_id])
                closest_origin = origin_id
        if closest_origin is not None:
            origin_by_node[node_id] = closest_origin

    return route_plan, origin_by_node


def build_instance_config(
    detail: dict[str, Any],
    summary_by_id: dict[str, dict[str, Any]],
    link_cache: dict[tuple[int, str], dict[str, Any]],
    placement_map: dict[str, list[int]],
    request_map: dict[str, list[dict[str, Any]]],
    step_entry: dict[str, Any],
    route_plan: dict[str, dict[str, dict[str, Any]]],
    origin_by_node: dict[str, str],
) -> dict[str, Any]:
    instance_id = detail['instance_id']
    otcp_node_id = detail.get('extra', {}).get('OTCPNodeId', instance_id)

    config_map: dict[str, Any] = {
        'instance_id': instance_id,
        'link_infos': {},
        'end_infos': {},
        'route_infos': {},
        'origin_node_id': origin_by_node.get(otcp_node_id, ''),
        'otcp': {
            'step': step_entry['step'],
            'timestamp_ms': step_entry['timestamp_ms'],
            'node_id': otcp_node_id,
            'method': step_entry.get('method', detail.get('extra', {}).get('OTCPMethod', 'otcp')),
            'original_type': detail.get('extra', {}).get('OTCPOriginalType', detail.get('type')),
            'cached_contents': placement_map.get(otcp_node_id, []),
            'pending_requests': request_map.get(otcp_node_id, []),
            'placement_index': placement_map,
        },
    }

    neighbor_addrs_by_node: dict[str, str] = {}

    for link_id, conn in (detail.get('connections') or {}).items():
        link_detail = link_cache.get((detail['node_index'], link_id)) or link_cache.get((conn['end_node_index'], link_id))
        if not link_detail:
            continue
        end_infos = link_detail.get('end_infos') or []
        address_infos = link_detail.get('address_infos') or []

        local_index = 0
        for index, end_info in enumerate(end_infos):
            if end_info.get('instance_id') == instance_id:
                local_index = index
                break
        remote_index = 1 - local_index if len(end_infos) > 1 else local_index

        remote_info = end_infos[remote_index]
        remote_summary = summary_by_id.get(remote_info['instance_id'], {})
        remote_address = address_infos[remote_index] if remote_index < len(address_infos) else {}

        config_map['link_infos'][link_id] = address_infos[local_index] if local_index < len(address_infos) else {}
        config_map['end_infos'][link_id] = {
            'instance_id': remote_info['instance_id'],
            'type': remote_info['instance_type'],
            'otcp_node_id': remote_summary.get('extra', {}).get('OTCPNodeId', remote_info['instance_id']),
            'v4_addr': remote_address.get('IPV4') if isinstance(remote_address, dict) else None,
        }
        remote_node_id = config_map['end_infos'][link_id]['otcp_node_id']
        remote_v4_addr = config_map['end_infos'][link_id]['v4_addr']
        if remote_node_id and remote_v4_addr:
            neighbor_addrs_by_node[remote_node_id] = remote_v4_addr

    for target_node_id, route_info in route_plan.get(otcp_node_id, {}).items():
        next_hop_node_id = route_info['next_hop_node_id']
        next_hop_addr = neighbor_addrs_by_node.get(next_hop_node_id)
        if not next_hop_addr:
            continue
        config_map['route_infos'][target_node_id] = {
            'next_hop_node_id': next_hop_node_id,
            'next_hop_v4_addr': next_hop_addr,
            'hop_count': route_info['hop_count'],
            'path_cost': route_info['path_cost'],
        }

    return config_map


def push_configs(base_url: str, manifest_path: Path, step: int | None, verify: bool) -> dict[str, Any]:
    manifest, entry, _, placement_payload, request_payload = load_step_artifacts(manifest_path, step)
    placement_map = placement_payload.get('placement', {})
    request_map = build_request_map(request_payload)

    summary_by_id, detail_by_id = build_instance_index(base_url)
    link_cache = build_link_detail_cache(base_url, detail_by_id)
    route_plan, origin_by_node = build_route_plan(summary_by_id, detail_by_id, link_cache)
    etcd_base_url = get_etcd_base_url(base_url)

    pushed_keys = []
    sample_config = None
    for instance_id, detail in detail_by_id.items():
        config_map = build_instance_config(detail, summary_by_id, link_cache, placement_map, request_map, entry, route_plan, origin_by_node)
        key = f"/instance_config/node_{detail['node_index']}/{instance_id}"
        value = json.dumps(config_map, sort_keys=True)
        put_etcd_value(etcd_base_url, key, value)
        pushed_keys.append(key)
        if sample_config is None:
            sample_config = {'key': key, 'value': config_map}

    verification = None
    if verify and pushed_keys:
        raw_value = get_etcd_value(etcd_base_url, pushed_keys[0])
        verification = {
            'key': pushed_keys[0],
            'found': raw_value is not None,
            'decoded': json.loads(raw_value) if raw_value else None,
        }

    return {
        'manifest_file': str(manifest_path),
        'sequence_dir': manifest.get('sequence_dir'),
        'step': entry['step'],
        'timestamp_ms': entry['timestamp_ms'],
        'pushed_instances': len(pushed_keys),
        'sample_config': sample_config,
        'verification': verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Push OTCP configs into OpenSN instance_config keys.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    payload = push_configs(args.base_url, args.manifest, args.step, args.verify)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
"""Replay an exported OpenSN sequence manifest and optionally collect runtime status.

Usage:
    python -m code.ns.opensn_apply_manifest \
        --manifest results/opensn_lce_lru_apply_smoke/sequence_manifest.json \
        --collect-runtime-status --verify-configs
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from .opensn_otcp_integration import DEFAULT_BASE_URL, DEFAULT_RUNTIME_STATUS_DIR, replay_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay an exported OpenSN manifest and collect runtime status.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--apply-steps', type=int, default=0)
    parser.add_argument('--reset-each-step', action='store_true')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--settle-seconds', type=float, default=6.0)
    parser.add_argument('--runtime-timeout', type=float, default=45.0)
    parser.add_argument('--runtime-status-dir', type=Path, default=DEFAULT_RUNTIME_STATUS_DIR)
    parser.add_argument('--verify-configs', action='store_true')
    parser.add_argument('--collect-runtime-status', action='store_true')
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    replay_args = Namespace(
        output_dir=args.manifest.parent,
        base_url=args.base_url,
        apply_steps=args.apply_steps,
        reset_each_step=args.reset_each_step,
        timeout=args.timeout,
        settle_seconds=args.settle_seconds,
        runtime_timeout=args.runtime_timeout,
        runtime_status_dir=args.runtime_status_dir,
        verify_configs=args.verify_configs,
        collect_runtime_status=args.collect_runtime_status,
    )

    payload = replay_sequence(replay_args, manifest, args.manifest.parent)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
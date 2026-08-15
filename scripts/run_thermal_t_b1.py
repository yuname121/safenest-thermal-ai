#!/usr/bin/env python3
"""Run the Thermal T-B1 Stage-1 preflight or the explicitly authorized later experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.thermal.t_b1_runner import (  # noqa: E402
    FULL_MODE,
    STAGE1_MODE,
    RunnerContractError,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeNest Thermal T-B1 controlled preprocessing runner")
    parser.add_argument("--canonical-root", default="", help="Configurable external canonical artifact root")
    parser.add_argument("--work-root", default="/tmp/safenest-thermal-t-b1", help="Local scratch/work root")
    parser.add_argument("--output-root", default="", help="Persistent result root")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--mode", choices=(STAGE1_MODE, FULL_MODE), default=STAGE1_MODE)
    parser.add_argument("--dry-run", action="store_true", help="Validate only; required for FULL_EXPERIMENT preflight")
    parser.add_argument("--execute", action="store_true", help="Execute fixture smoke or explicitly authorized later experiment")
    parser.add_argument("--authorize-full-experiment", action="store_true", help="Owner authorization gate for later FULL_EXPERIMENT")
    parser.add_argument("--fixture-smoke", action="store_true", help="Run tiny fixture-only smoke training; never project evidence")
    args = parser.parse_args()
    dry_run = bool(args.dry_run)
    execute = bool(args.execute)
    if not dry_run and not execute:
        dry_run = True
    try:
        result = run(
            mode=args.mode,
            canonical_root=args.canonical_root,
            work_root=args.work_root,
            output_root=args.output_root,
            repo_root=args.repo_root,
            dry_run=dry_run,
            execute=execute,
            owner_authorized=bool(args.authorize_full_experiment),
            fixture_smoke=bool(args.fixture_smoke),
        )
    except (RunnerContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "phase": "T-B1", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

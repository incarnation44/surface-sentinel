"""CLI skeleton for Surface Sentinel.

Only ``--version`` and ``doctor`` are implemented in the SS-DESIGN-1.1
scaffold. Pipeline, data, evaluation, UI, and agent commands are added by
later tasks.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import CONFIG_SCHEMA_VERSION
from .version import __version__


def _doctor() -> dict[str, Any]:
    info: dict[str, Any] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_schema": CONFIG_SCHEMA_VERSION,
        "cwd": str(Path.cwd()),
        "opencv": None,
        "numpy": None,
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except Exception:
        info["opencv"] = None
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:
        info["numpy"] = None
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-sentinel",
        description="Deterministic surface-defect inspection and verification.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="print environment diagnostics as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(_doctor(), ensure_ascii=False, indent=2))
        return 0
    parser.print_help(sys.stdout)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

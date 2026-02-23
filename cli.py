"""Command-line helpers for llm_orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import scaffold_default_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-orchestrator-init",
        description="Generate a starter orchestrator.yaml in your current project.",
    )
    parser.add_argument(
        "--path",
        default="orchestrator.yaml",
        help="Output path for generated config template (default: orchestrator.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite target file if it already exists.",
    )

    args = parser.parse_args()
    output = scaffold_default_config(target_path=args.path, overwrite=args.force)

    print(f"Generated config template: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
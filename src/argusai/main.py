"""
ArgusAI CLI entrypoint.
"""

from __future__ import annotations
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*validate_default.*Field.*no effect.*",
)

from .cli import run_cli


def main() -> int:
    """Run ArgusAI CLI."""
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
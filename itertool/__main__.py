"""CLI entry point for Iter."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from unified_ai_client import silence_sdks
from itertool.processor import run_iter
from itertool.models import Preset
from itertool.config import AppConfig


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the run subcommand.

    Args:
        args: Parsed argument namespace.
    """
    try:
        config = AppConfig.load()

        if args.preset:
            preset_path = Path("Presets") / f"{args.preset}.json"
            preset = Preset.load(preset_path)
        else:
            preset = Preset()

        # CLI flags override preset values when explicitly provided
        if args.node_mode:
            preset.node_mode = args.node_mode
        if args.preprocessing_prompt:
            preset.preprocessing_prompt = args.preprocessing_prompt
        if args.node_prompt:
            preset.node_prompt = args.node_prompt
        if args.postprocessing_prompt:
            preset.postprocessing_prompt = args.postprocessing_prompt

        run_iter(
            input_file=args.input,
            output_file=args.output,
            preset=preset,
            config=config,
        )
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_mcp(args: argparse.Namespace) -> None:
    """Handle the mcp subcommand -- start the FastMCP server on stdio.

    Args:
        args: Parsed argument namespace (unused).
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("'mcp' library not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)
    from itertool.mcp_tools import register_tools
    mcp = FastMCP("Iter")
    register_tools(mcp)
    mcp.run()


def cmd_gui(args: argparse.Namespace) -> None:
    """Handle the gui subcommand -- launch the Streamlit web interface.

    Args:
        args: Parsed argument namespace (unused).
    """
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "gui" / "app.py"),
    ])


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with subcommands.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="iter",
        description="Iter — Iterative node-based document processor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Process a Markdown document node by node.")
    p_run.add_argument("--input", required=True, help="Input Markdown file path.")
    p_run.add_argument("--output", required=True, help="Output file path.")
    p_run.add_argument("--preset", default=None, help="Preset name to load from Presets/ directory.")
    p_run.add_argument(
        "--node-mode",
        choices=["paragraph", "section"],
        default=None,
        help="Node granularity: 'paragraph' or 'section'.",
    )
    p_run.add_argument("--preprocessing-prompt", default=None, help="Preprocessing prompt text.")
    p_run.add_argument("--node-prompt", default=None, help="Node prompt text.")
    p_run.add_argument("--postprocessing-prompt", default=None, help="Postprocessing prompt text.")
    p_run.set_defaults(func=cmd_run)

    p_mcp = subparsers.add_parser("mcp", help="Start the MCP server on stdio.")
    p_mcp.set_defaults(func=cmd_mcp)

    p_gui = subparsers.add_parser("gui", help="Launch the Streamlit web interface.")
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main() -> None:
    """Main entry point for the Iter CLI."""
    silence_sdks()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

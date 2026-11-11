"""CLI entry point for Iter."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from unified_ai_client import silence_sdks
from itertool.checkpoint import CheckpointMismatchError
from itertool.output import default_output_path
from itertool.processor import run_iter
from itertool.models import Preset
from itertool.config import AppConfig
import itertool.logger as logger


_CHOICES = {"c": "continue", "r": "restart"}


def _ask_mismatch_policy(exc: CheckpointMismatchError) -> str:
    """Ask the user what to do with a progress file from a different run.

    Args:
        exc: The mismatch, whose message names what changed.

    Returns:
        'continue' or 'restart'.

    Raises:
        CheckpointMismatchError: If there is no one to ask (stdin closed).
    """
    logger.log_warning(str(exc))
    while True:
        try:
            answer = input("Continue with the saved results, or restart from scratch? [c/r]: ")
        except EOFError:
            raise exc from None
        policy = _CHOICES.get(answer.strip().lower()[:1])
        if policy:
            return policy


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the run subcommand.

    Args:
        args: Parsed argument namespace.
    """
    try:
        config = AppConfig.load()
        config.apply_to_client()

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

        run_kwargs = dict(
            input_path=args.input,
            output_path=args.output or default_output_path(args.input, preset),
            preset=preset,
            config=config,
            context_file=args.context,
        )
        try:
            run_iter(**run_kwargs)
        except CheckpointMismatchError as exc:
            run_iter(**run_kwargs, on_mismatch=_ask_mismatch_policy(exc))
    except KeyboardInterrupt:
        logger.log_warning("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        logger.log_error(f"Fatal error: {exc}")
        sys.exit(1)


def cmd_mcp(args: argparse.Namespace) -> None:
    """Handle the mcp subcommand -- start the FastMCP server on stdio.

    Args:
        args: Parsed argument namespace (unused).
    """
    # stdout carries the JSON-RPC stream - the mcp backend keeps the identical
    # output format but routes it to stderr, where it cannot corrupt the framing.
    # Selected before anything can log, so no message can reach stdout.
    logger.set_backend("mcp")

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.log_error("'mcp' library not installed. Run: pip install mcp")
        sys.exit(1)
    from itertool.mcp_tools import register_tools

    mcp = FastMCP("Iter")
    register_tools(mcp)
    mcp.run()


def cmd_gui(args: argparse.Namespace) -> None:
    """Handle the gui subcommand -- launch the Streamlit web interface.

    Args:
        args: Parsed argument namespace; ``args.input``, when given, starts as the input path.
    """
    command = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "gui" / "app.py")]
    if args.input:
        # Streamlit hands what follows "--" to the script as its sys.argv.
        command += ["--", "--input", str(Path(args.input).resolve())]
    try:
        subprocess.run(command)
    except Exception as exc:
        logger.log_error(f"Failed to launch the GUI: {exc}")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with subcommands.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="iter",
        description="Iter - Iterative node-based document processor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Process a Markdown document, or a folder of them, node by node.")
    p_run.add_argument("--input", required=True, help="Input Markdown file, or a folder of Markdown files.")
    p_run.add_argument(
        "--output",
        default=None,
        help=(
            "Output file, or output folder for a folder input in the mirror layout. "
            "Default: <name>_processed<ext> next to a file; <folder>_processed beside a folder "
            "(mirror), or <folder>_processed.md (merged)."
        ),
    )
    p_run.add_argument("--preset", default=None, help="Preset name to load from Presets/ directory.")
    p_run.add_argument(
        "--context",
        default=None,
        help="Reference context file, appended to the system prompt of every AI call.",
    )
    p_run.add_argument(
        "--node-mode",
        choices=["paragraph", "section", "file"],
        default=None,
        help="Node granularity: 'paragraph', 'section' or 'file'.",
    )
    p_run.add_argument("--preprocessing-prompt", default=None, help="Preprocessing prompt text.")
    p_run.add_argument("--node-prompt", default=None, help="Node prompt text.")
    p_run.add_argument("--postprocessing-prompt", default=None, help="Postprocessing prompt text.")
    p_run.set_defaults(func=cmd_run)

    p_mcp = subparsers.add_parser("mcp", help="Start the MCP server on stdio.")
    p_mcp.set_defaults(func=cmd_mcp)

    p_gui = subparsers.add_parser("gui", help="Launch the Streamlit web interface.")
    p_gui.add_argument("input", nargs="?", help="Markdown file or folder to set as the input.")
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

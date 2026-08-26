"""
Shared CLI argument parsing for all phase scripts.

Usage at the very top of a phase script (BEFORE importing config):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.cli import parse_mode
    parse_mode()          # sets FINCAUSAL_DEMO env var based on --mode

    import config          # now picks up the right mode
    config.mode_banner()

This has to happen before `import config` because config.py reads the
FINCAUSAL_DEMO environment variable at import time.
"""

import argparse
import os


def parse_mode(extra_args=None):
    """
    Parses --mode {demo,full} (default: demo) and sets FINCAUSAL_DEMO
    accordingly. Returns the parsed argparse.Namespace so callers can add
    their own arguments via `extra_args` (a callable that receives the
    parser and can call parser.add_argument(...)).
    """
    parser = argparse.ArgumentParser(
        description="FinCausalSAE phase script. Use --mode demo (default) "
                     "to run on CPU with a tiny GPT-2 model and synthetic "
                     "data, or --mode full for the real Llama-3.1-8B "
                     "research pipeline (requires a GPU)."
    )
    parser.add_argument(
        "--mode", choices=["demo", "full"], default="demo",
        help="demo = CPU-friendly smoke test (default). "
             "full = real research run, requires GPU."
    )
    if extra_args is not None:
        extra_args(parser)

    args, _unknown = parser.parse_known_args()
    os.environ["FINCAUSAL_DEMO"] = "0" if args.mode == "full" else "1"
    return args

#!/usr/bin/env python3
"""Thin entry point so `python orchestrator.py ...` keeps working.

The real CLI lives in orchestrator/cli.py (also exposed as the `rayllm`
console script when installed via pip).
"""

import sys

from orchestrator.cli import main

if __name__ == "__main__":
    sys.exit(main())

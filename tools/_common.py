"""Shared utilities for CLI tools.

Provides consistent JSON output formatting and error handling
so all tools follow the same interface contract.
"""

import json
import os
import sys

# Load .env if python-dotenv is available (for standalone invocation)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass


def output(data):
    """Print JSON data to stdout and exit 0."""
    json.dump(data, sys.stdout, indent=2, default=str)
    print()  # trailing newline
    sys.exit(0)


def error(message, details=None, exit_code=1):
    """Print JSON error to stderr and exit non-zero."""
    err = {"error": message}
    if details is not None:
        err["details"] = details
    json.dump(err, sys.stderr, indent=2, default=str)
    print(file=sys.stderr)
    sys.exit(exit_code)

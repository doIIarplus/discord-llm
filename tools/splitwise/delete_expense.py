#!/usr/bin/env python3
"""Delete a Splitwise expense by ID."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expense_id", type=int, help="The expense ID to delete")
    args = parser.parse_args()

    client = SplitwiseClient()
    result = client.post(f"/delete_expense/{args.expense_id}", {})

    if result.get("success"):
        output({"deleted": True, "expense_id": args.expense_id})
    else:
        error(
            "Failed to delete expense",
            details={"expense_id": args.expense_id, "response": result},
        )


if __name__ == "__main__":
    main()

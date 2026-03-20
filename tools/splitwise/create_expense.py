#!/usr/bin/env python3
"""Create a Splitwise expense.

Supports three split modes:
  - Equal split (default): total divided evenly among all participants
  - Ratio split (--ratios): proportional split by integer ratios
  - Custom shares (--shares): exact dollar amounts per person

The payer is always the authenticated user unless --paid-by is specified.
The payer is automatically included in the split. --split-with lists the
OTHER participants (friends' Splitwise user IDs).

Examples:
  # $60 dinner split equally between you and two friends
  create_expense.py --amount 60 --description "Dinner" --split-with 12345 67890

  # $100 groceries, you pay 50%, friend A pays 30%, friend B pays 20%
  create_expense.py --amount 100 --description "Groceries" --split-with 12345 67890 --ratios 5 3 2

  # $75 with exact per-person amounts (payer first, then friends in order)
  create_expense.py --amount 75 --description "Supplies" --split-with 12345 67890 --shares 25.00 30.00 20.00
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output, error


def _check_duplicate(client, description, amount):
    """Check if an expense with the same description and amount was created in the last 60 seconds."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
    result = client.get(f"/get_expenses?limit=5&dated_after={cutoff}")
    for exp in result.get("expenses", []):
        if exp.get("deleted_at"):
            continue
        if (exp.get("description", "").strip().lower() == description.strip().lower()
                and exp.get("cost") == str(round(amount, 2))):
            json.dump({"error": f"Duplicate expense detected (id: {exp['id']}), skipping creation"}, sys.stdout)
            print()
            sys.exit(1)


def _check_duplicate(client, description, amount):
    """Check if a matching expense was created in the last 60 seconds."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
    result = client.get(f"/get_expenses?limit=5&dated_after={cutoff}")
    for exp in result.get("expenses", []):
        if exp.get("deleted_at"):
            continue
        if (exp.get("description", "").strip().lower() == description.strip().lower()
                and float(exp.get("cost", 0)) == round(amount, 2)):
            json.dump({"error": f"Duplicate expense detected (id: {exp['id']}), skipping creation"}, sys.stdout)
            print()
            sys.exit(1)


def _build_equal_split(amount, payer_id, friend_ids, group_id):
    """Build form data for an equal split."""
    all_ids = [payer_id] + friend_ids
    num_people = len(all_ids)
    share = round(amount / num_people, 2)
    # Adjust payer's owed share to absorb rounding
    payer_owed = round(amount - share * len(friend_ids), 2)

    data = {
        "cost": str(round(amount, 2)),
        "description": "",  # filled by caller
        "currency_code": "USD",
        "split_equally": False,
        "users__0__user_id": payer_id,
        "users__0__paid_share": str(round(amount, 2)),
        "users__0__owed_share": str(payer_owed),
    }
    for i, fid in enumerate(friend_ids, start=1):
        data[f"users__{i}__user_id"] = fid
        data[f"users__{i}__paid_share"] = "0.00"
        data[f"users__{i}__owed_share"] = str(share)

    return data


def _build_ratio_split(amount, payer_id, friend_ids, ratios, group_id):
    """Build form data for a ratio-based split."""
    all_ids = [payer_id] + friend_ids
    if len(ratios) != len(all_ids):
        error(
            f"Number of ratios ({len(ratios)}) must match number of "
            f"participants ({len(all_ids)}): payer + {len(friend_ids)} friends"
        )

    total_ratio = sum(ratios)
    shares = [round(amount * r / total_ratio, 2) for r in ratios]
    # Fix rounding: adjust the largest share
    diff = round(amount - sum(shares), 2)
    if diff != 0:
        max_idx = shares.index(max(shares))
        shares[max_idx] = round(shares[max_idx] + diff, 2)

    data = {
        "cost": str(round(amount, 2)),
        "description": "",
        "currency_code": "USD",
        "split_equally": False,
    }
    for i, (uid, share) in enumerate(zip(all_ids, shares)):
        data[f"users__{i}__user_id"] = uid
        data[f"users__{i}__paid_share"] = str(round(amount, 2)) if uid == payer_id else "0.00"
        data[f"users__{i}__owed_share"] = str(share)

    return data


def _build_custom_split(amount, payer_id, friend_ids, shares_list, group_id):
    """Build form data for exact per-person amounts."""
    all_ids = [payer_id] + friend_ids
    if len(shares_list) != len(all_ids):
        error(
            f"Number of shares ({len(shares_list)}) must match number of "
            f"participants ({len(all_ids)}): payer + {len(friend_ids)} friends"
        )

    shares_total = round(sum(shares_list), 2)
    if abs(shares_total - round(amount, 2)) > 0.01:
        error(
            f"Shares sum ({shares_total}) does not match amount ({round(amount, 2)})"
        )

    data = {
        "cost": str(round(amount, 2)),
        "description": "",
        "currency_code": "USD",
        "split_equally": False,
    }
    for i, (uid, share) in enumerate(zip(all_ids, shares_list)):
        data[f"users__{i}__user_id"] = uid
        data[f"users__{i}__paid_share"] = str(round(amount, 2)) if uid == payer_id else "0.00"
        data[f"users__{i}__owed_share"] = str(round(share, 2))

    return data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--amount", type=float, required=True,
                        help="Total expense amount")
    parser.add_argument("--description", required=True,
                        help="What the expense is for")
    parser.add_argument("--split-with", type=int, nargs="+", required=True,
                        metavar="USER_ID",
                        help="Splitwise user IDs of friends to split with")
    parser.add_argument("--ratios", type=int, nargs="+",
                        metavar="RATIO",
                        help="Integer ratios for split (payer first, then friends in order)")
    parser.add_argument("--shares", type=float, nargs="+",
                        metavar="AMOUNT",
                        help="Exact dollar amounts per person (payer first, then friends)")
    parser.add_argument("--group-id", type=int, default=0,
                        help="Splitwise group ID (optional)")
    parser.add_argument("--currency", default="USD",
                        help="Currency code (default: USD)")
    parser.add_argument("--paid-by", type=int, default=None,
                        metavar="USER_ID",
                        help="User ID of payer (default: authenticated user)")
    args = parser.parse_args()

    if args.ratios and args.shares:
        error("Cannot specify both --ratios and --shares")

    client = SplitwiseClient()

    # Resolve payer
    if args.paid_by:
        payer_id = args.paid_by
    else:
        user = client.get("/get_current_user").get("user", {})
        payer_id = user["id"]

    friend_ids = args.split_with

    # Build the expense data
    if args.shares:
        data = _build_custom_split(args.amount, payer_id, friend_ids, args.shares, args.group_id)
    elif args.ratios:
        data = _build_ratio_split(args.amount, payer_id, friend_ids, args.ratios, args.group_id)
    else:
        data = _build_equal_split(args.amount, payer_id, friend_ids, args.group_id)

    # Fill in common fields
    data["description"] = args.description
    data["currency_code"] = args.currency
    if args.group_id:
        data["group_id"] = args.group_id

    # Check for duplicate before creating
    _check_duplicate(client, args.description, args.amount)

    # Check for duplicate expense created in the last 60 seconds
    _check_duplicate(client, args.description, args.amount)

    result = client.post_form("/create_expense", data)
    expenses = result.get("expenses", [])

    if expenses:
        exp = expenses[0]
        output({
            "created": True,
            "expense_id": exp.get("id"),
            "description": exp.get("description"),
            "cost": exp.get("cost"),
            "currency": exp.get("currency_code"),
            "date": exp.get("date"),
            "users": [
                {
                    "user_id": u.get("user", {}).get("id"),
                    "name": f"{u.get('user', {}).get('first_name', '')} {u.get('user', {}).get('last_name', '')}".strip(),
                    "paid_share": u.get("paid_share"),
                    "owed_share": u.get("owed_share"),
                }
                for u in exp.get("users", [])
            ],
        })
    else:
        error("Expense creation returned no data", details=result)


if __name__ == "__main__":
    main()

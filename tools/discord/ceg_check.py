#!/usr/bin/env python3
"""Check CEG stock price vs reference and send a roast/praise message to donairplus."""

import os
import subprocess
import sys

REFERENCE_PRICE = 252.0
CHANNEL_ID = "1274590758216925316"
DONAIRPLUS = "<@132687001257246720>"


def get_ceg_price():
    import yfinance as yf
    ticker = yf.Ticker("CEG")
    return ticker.fast_info.last_price


def build_message(price: float) -> str:
    pct = (price - REFERENCE_PRICE) / REFERENCE_PRICE * 100
    if price < REFERENCE_PRICE:
        return (
            f"hey {DONAIRPLUS} your CEG play aged like milk lmaooo "
            f"you're down {abs(pct):.1f}% on your 100x leverage, "
            f"i hope you got liquidated"
        )
    else:
        return (
            f"ok {DONAIRPLUS} fine you were right, CEG is up {pct:.1f}% "
            f"since you bought, i was wrong, this is humiliating"
        )


def send_message(content: str):
    script = os.path.join(os.path.dirname(__file__), "send_message.py")
    subprocess.run(
        [sys.executable, script, "--channel-id", CHANNEL_ID, "--content", content],
        check=True,
    )


if __name__ == "__main__":
    price = get_ceg_price()
    msg = build_message(price)
    print(f"CEG price: ${price:.2f} (ref ${REFERENCE_PRICE})")
    print(f"Sending: {msg}")
    send_message(msg)

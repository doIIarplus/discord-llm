"""Watch a Gmail inbox via IMAP IDLE for Google Fi monthly statements,
auto-create a Splitwise expense, and DM the user the result.

Account / split are hardcoded for jaspershan's plan:
  payer:        jaspershan (17073080)
  group:        Google Fi (33169167)
  split:        ratios 2:1:1 (jasper 2 lines, junshu 1, Yang 1)
  currency:     USD
"""

import asyncio
import email
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

from aioimaplib import aioimaplib

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.gmail_fi_billing")

# --- Fixed config ----------------------------------------------------------
DM_USER_ID = 118567805678256128
SUBJECT_FILTER = "Google Fi monthly statement"

SPLITWISE_GROUP_ID = 33169167
SPLITWISE_PAYER_ID = 17073080            # jaspershan
SPLITWISE_SPLIT_WITH = ["43460287", "51458857"]  # junshu, Yang
SPLITWISE_RATIOS = ["2", "1", "1"]       # payer first, then split-with order
SPLITWISE_CURRENCY = "USD"

CREATE_EXPENSE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "splitwise", "create_expense.py",
)

# --- IMAP ------------------------------------------------------------------
IMAP_HOST = "imap.gmail.com"
IDLE_TIMEOUT_SECONDS = 25 * 60
RECONNECT_BACKOFF_INITIAL = 5
RECONNECT_BACKOFF_MAX = 300

TOTAL_RE = re.compile(r'Your total is \$([\d,]+\.\d{2})')
STATEMENT_DATE_RE = re.compile(r'summary of your ([A-Z][a-z]+ \d+) statement')

# Month name → 1-based number
_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}


def _extract_body_text(raw_bytes: bytes) -> str:
    """Pull all text/plain bodies out of an email."""
    msg = email.message_from_bytes(raw_bytes)
    parts = []
    for part in msg.walk():
        if part.get_content_type() != "text/plain":
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            parts.append(payload.decode(charset, errors="replace"))
        except Exception:
            continue
    return "\n".join(parts)


def _parse_statement(raw_bytes: bytes) -> dict | None:
    """Return {'amount': float, 'date': 'YYYY-MM-DD'} or None if either is missing."""
    msg = email.message_from_bytes(raw_bytes)
    body = _extract_body_text(raw_bytes)
    m_total = TOTAL_RE.search(body)
    if not m_total:
        return None
    amount = float(m_total.group(1).replace(",", ""))

    m_date = STATEMENT_DATE_RE.search(body)
    if not m_date:
        return None
    month_name, day = m_date.group(1).split()
    month_num = _MONTHS.get(month_name)
    if month_num is None:
        return None

    # Year comes from the email's Date header — statement month/day from body
    try:
        sent = email.utils.parsedate_to_datetime(msg.get("Date", ""))
        year = sent.year
    except Exception:
        year = datetime.now().year

    return {
        "amount": amount,
        "date": f"{year}-{month_num:02d}-{int(day):02d}",
        "month_label": f"{month_name[:3]} {year}",
    }


class GmailFiBillingPlugin(BasePlugin):
    name = "gmail_fi_billing"
    version = "1.0.0"
    description = "Auto-create Splitwise expense from Google Fi monthly statements"

    async def on_load(self):
        self._email = os.getenv("GMAIL_FI_ADDRESS", "")
        self._password = os.getenv("GMAIL_FI_APP_PASSWORD", "")
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

        if not self._email or not self._password:
            self.logger.warning(
                "GMAIL_FI_ADDRESS / GMAIL_FI_APP_PASSWORD not set — plugin idle"
            )
            return

        self._task = asyncio.create_task(self._watch_loop())

    async def on_unload(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def self_test(self) -> bool:
        return True

    # ── DM helper ────────────────────────────────────────────────────────

    async def _dm(self, content: str):
        try:
            user = await self.ctx.discord_client.fetch_user(DM_USER_ID)
            await user.send(content)
        except Exception as e:
            self.logger.warning(f"Failed to DM user {DM_USER_ID}: {e}")

    # ── Main loop ────────────────────────────────────────────────────────

    async def _watch_loop(self):
        backoff = RECONNECT_BACKOFF_INITIAL
        while not self._stop.is_set():
            imap = None
            try:
                self.logger.info(f"Connecting to {IMAP_HOST} as {self._email}")
                imap = aioimaplib.IMAP4_SSL(host=IMAP_HOST)
                await imap.wait_hello_from_server()
                resp = await imap.login(self._email, self._password)
                if resp.result != "OK":
                    raise RuntimeError(f"IMAP login failed: {resp}")
                await imap.select("INBOX")
                self.logger.info("IMAP connected, inbox selected")
                backoff = RECONNECT_BACKOFF_INITIAL

                # Catch up on anything unread (e.g. bot was offline when a bill arrived)
                await self._scan_and_process(imap)

                while not self._stop.is_set():
                    idle_task = await imap.idle_start(timeout=IDLE_TIMEOUT_SECONDS)
                    try:
                        await imap.wait_server_push()
                    except asyncio.TimeoutError:
                        pass
                    imap.idle_done()
                    try:
                        await asyncio.wait_for(idle_task, timeout=10)
                    except asyncio.TimeoutError:
                        self.logger.warning("IDLE didn't acknowledge DONE — reconnecting")
                        break
                    await self._scan_and_process(imap)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"IMAP loop error: {e}; reconnecting in {backoff}s")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
            finally:
                if imap is not None:
                    try:
                        await imap.logout()
                    except Exception:
                        pass

    # ── Email handling ───────────────────────────────────────────────────

    async def _scan_and_process(self, imap):
        resp = await imap.uid_search(
            "UNSEEN", "SUBJECT", f'"{SUBJECT_FILTER}"'
        )
        if resp.result != "OK":
            self.logger.warning(f"uid_search failed: {resp}")
            return
        ids_line = resp.lines[0] if resp.lines else b""
        uids = ids_line.decode().split() if ids_line else []
        if not uids:
            return

        self.logger.info(f"Found {len(uids)} unread Fi statement(s): {uids}")
        for uid in uids:
            try:
                await self._process_one(imap, uid)
            except Exception as e:
                self.logger.exception(f"Failed to process uid={uid}: {e}")

    async def _process_one(self, imap, uid: str):
        fetch_resp = await imap.uid("fetch", uid, "(RFC822)")
        if fetch_resp.result != "OK":
            self.logger.warning(f"fetch uid={uid} failed: {fetch_resp}")
            return
        raw = max(
            (bytes(ln) for ln in fetch_resp.lines if isinstance(ln, (bytes, bytearray))),
            key=len, default=b"",
        )
        if not raw:
            self.logger.warning(f"uid={uid}: empty RFC822 payload")
            return

        parsed = _parse_statement(raw)
        if parsed is None:
            self.logger.warning(f"uid={uid}: couldn't parse amount/date from body")
            await self._dm(
                f"⚠️ Fi statement (uid {uid}) — couldn't parse amount/date. "
                f"check the email manually and create the Splitwise expense by hand"
            )
            # Mark read so we don't loop on the same broken email
            await imap.uid("store", uid, "+FLAGS", "(\\Seen)")
            return

        amount = parsed["amount"]
        date = parsed["date"]
        label = parsed["month_label"]
        desc = f"Google Fi {label}"

        self.logger.info(
            f"uid={uid}: creating Splitwise expense '{desc}' ${amount:.2f} dated {date}"
        )

        cmd = [
            sys.executable, CREATE_EXPENSE_SCRIPT,
            "--amount", str(amount),
            "--description", desc,
            "--date", date,
            "--paid-by", str(SPLITWISE_PAYER_ID),
            "--group-id", str(SPLITWISE_GROUP_ID),
            "--currency", SPLITWISE_CURRENCY,
            "--split-with", *SPLITWISE_SPLIT_WITH,
            "--ratios", *SPLITWISE_RATIOS,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = (stderr.decode("utf-8", errors="replace") or
                   stdout.decode("utf-8", errors="replace"))[:500]
            self.logger.error(f"uid={uid}: create_expense failed: {err}")
            await self._dm(
                f"❌ Fi statement {label} (${amount:.2f}) — Splitwise create FAILED:\n"
                f"```\n{err}\n```\n"
                f"email left unread, will retry on next reconnect"
            )
            return

        # Success — parse result and DM confirmation
        try:
            import json
            result = json.loads(stdout)
            exp_id = result.get("expense_id")
            per_user = "\n".join(
                f"  • {u['name'].strip()}: owes ${u['owed_share']}"
                for u in result.get("users", [])
                if float(u.get("owed_share") or 0) > 0
            )
        except Exception:
            exp_id = "?"
            per_user = "(couldn't parse split details from response)"

        await self._dm(
            f"✅ Google Fi {label} — **${amount:.2f}** (id {exp_id}, dated {date})\n"
            f"{per_user}"
        )

        # Mark read so we don't reprocess on next scan
        await imap.uid("store", uid, "+FLAGS", "(\\Seen)")
        self.logger.info(f"uid={uid}: processed and marked read")

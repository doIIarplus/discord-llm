"""Watch a Gmail inbox via IMAP IDLE for Google Fi monthly statements,
auto-create a Splitwise expense, and DM the user the result.

Emails are NEVER deleted or marked Seen — they stay visible in your inbox.
Dedupe is handled locally via [[email_dedupe]] (a SQLite store of
Message-IDs we've already processed).

On first run (empty dedupe store), every existing matching email is treated
as already-processed. Only statements that arrive AFTER first startup are acted on.

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

# Ensure project root is on sys.path so we can import email_dedupe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import email_dedupe  # noqa: E402
from plugin_base import BasePlugin  # noqa: E402

logger = logging.getLogger("Plugin.gmail_fi_billing")

PLUGIN_NAME = "gmail_fi_billing"

# --- Fixed config ----------------------------------------------------------
DM_USER_ID = 118567805678256128
SUBJECT_FILTER = "Google Fi monthly statement"

SPLITWISE_GROUP_ID = 33169167
SPLITWISE_PAYER_ID = 17073080            # jaspershan
SPLITWISE_SPLIT_WITH = ["43460287", "51458857"]  # junshu, Yang
SPLITWISE_RATIOS = ["2", "1", "1"]
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
_MSGID_RE = re.compile(r'^Message-ID:\s*(.+?)\s*$', re.MULTILINE | re.IGNORECASE)

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}


def _join_fetch_bytes(lines) -> bytes:
    return b"\n".join(bytes(ln) for ln in lines if isinstance(ln, (bytes, bytearray)))


def _extract_message_id(raw_bytes: bytes) -> str:
    if not raw_bytes:
        return ""
    text = raw_bytes.decode("utf-8", errors="replace")
    m = _MSGID_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_body_text(raw_bytes: bytes) -> str:
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
    """Return {'amount', 'date', 'month_label'} or None if either is missing."""
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
    version = "2.0.0"
    description = "Auto-create Splitwise expense from Google Fi statements (no email deletion, local dedupe)"

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
        """Search ALL matching messages (any flag state), skip already-processed."""
        resp = await imap.uid_search("SUBJECT", f'"{SUBJECT_FILTER}"')
        if resp.result != "OK":
            self.logger.warning(f"uid_search failed: {resp}")
            return
        ids_line = resp.lines[0] if resp.lines else b""
        uids = ids_line.decode().split() if ids_line else []
        if not uids:
            return

        has_history = await asyncio.to_thread(email_dedupe.has_any, PLUGIN_NAME)
        if not has_history:
            seeded = await self._seed_backlog(imap, uids)
            self.logger.info(
                f"First run — seeded {seeded} existing matching email(s) as processed; "
                f"backlog ignored"
            )
            return

        self.logger.info(f"Found {len(uids)} matching Fi statement email(s) to inspect")
        for uid in uids:
            try:
                await self._process_one(imap, uid)
            except Exception as e:
                self.logger.exception(f"Failed to process uid={uid}: {e}")

    async def _seed_backlog(self, imap, uids) -> int:
        msg_ids: list[str] = []
        for uid in uids:
            r = await imap.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
            if r.result != "OK":
                continue
            mid = _extract_message_id(_join_fetch_bytes(r.lines))
            if mid:
                msg_ids.append(mid)
        return await asyncio.to_thread(
            email_dedupe.bulk_mark_processed, PLUGIN_NAME, msg_ids
        )

    async def _process_one(self, imap, uid: str):
        # Cheap dedupe check using just the Message-ID header.
        r = await imap.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
        if r.result != "OK":
            self.logger.warning(f"header-fetch uid={uid} failed: {r}")
            return
        mid = _extract_message_id(_join_fetch_bytes(r.lines))
        if mid and await asyncio.to_thread(email_dedupe.is_processed, PLUGIN_NAME, mid):
            return

        # Fetch full body WITHOUT setting \Seen.
        fr = await imap.uid("fetch", uid, "(BODY.PEEK[])")
        if fr.result != "OK":
            self.logger.warning(f"body-fetch uid={uid} failed: {fr}")
            return
        raw = b""
        for ln in fr.lines:
            if isinstance(ln, (bytes, bytearray)) and len(ln) > len(raw):
                raw = bytes(ln)
        if not raw or len(raw) < 100:
            self.logger.warning(f"uid={uid}: empty/short body, skipping")
            return

        if not mid:
            mid = _extract_message_id(raw)

        parsed = _parse_statement(raw)
        if parsed is None:
            self.logger.warning(f"uid={uid}: couldn't parse amount/date from body")
            await self._dm(
                f"⚠️ Fi statement (uid {uid}) — couldn't parse amount/date. "
                f"check the email manually and create the Splitwise expense by hand"
            )
            # Don't mark processed — we may want to retry after a parser fix.
            return

        amount = parsed["amount"]
        date = parsed["date"]
        label = parsed["month_label"]
        desc = f"Google Fi {label}"

        self.logger.info(
            f"uid={uid} mid={mid[:60]}: creating Splitwise expense "
            f"'{desc}' ${amount:.2f} dated {date}"
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
                f"not marked processed — will retry on next IDLE wake"
            )
            return

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

        if mid:
            await asyncio.to_thread(email_dedupe.mark_processed, PLUGIN_NAME, mid)
            self.logger.info(f"uid={uid}: processed and recorded in dedupe store")
        else:
            self.logger.warning(
                f"uid={uid}: created expense but no Message-ID found — "
                f"future scans may re-create this expense"
            )

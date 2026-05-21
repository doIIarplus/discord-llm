"""Watch a Gmail inbox via IMAP IDLE for Claude.ai magic-link emails
and forward the link to a Discord channel.

Emails are NEVER deleted or marked Seen — they stay visible in your inbox.
Dedupe is handled locally via [[email_dedupe]] (a SQLite store of
Message-IDs we've already processed).

On first run (empty dedupe store), every existing matching email is treated
as already-processed. Only emails that arrive AFTER first startup are acted on.
"""

import asyncio
import email
import logging
import os
import re
import sys

from aioimaplib import aioimaplib

# Ensure project root is on sys.path so we can import email_dedupe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import email_dedupe  # noqa: E402
from plugin_base import BasePlugin  # noqa: E402

logger = logging.getLogger("Plugin.gmail_magic_link")

PLUGIN_NAME = "gmail_magic_link"
TARGET_CHANNEL_ID = 1483701977954123806
SUBJECT_FILTER = "Secure link to log in to Claude.ai"
# Match any URL that contains "magic-link" anywhere — Anthropic keeps
# changing the path/query format and this future-proofs against further changes.
MAGIC_LINK_RE = re.compile(r'https?://[^\s"\'<>]*magic-link[^\s"\'<>]*')

IMAP_HOST = "imap.gmail.com"
IDLE_TIMEOUT_SECONDS = 25 * 60
# Backoff schedule for IMAP reconnection (seconds). Each entry is used for the
# nth consecutive failure; once we run off the end we stay at the last value.
RECONNECT_BACKOFF_SCHEDULE = (5, 15, 30, 60)
# Trigger reconnection if this many UID fetches fail in a row inside one
# _scan_and_process pass — a strong signal the underlying socket is dead even
# though the per-uid try/except is masking the IMAP-level errors.
MAX_CONSECUTIVE_FETCH_FAILURES = 3
# Hard ceiling on how long a single IDLE wait can block before we tear down
# and reconnect. The aioimaplib IDLE timeout doesn't always fire when the
# socket is half-dead, so we layer wait_for on top as a safety net.
IDLE_WAIT_TIMEOUT_SECONDS = IDLE_TIMEOUT_SECONDS + 60

# Match Message-ID: across any line of an IMAP fetch response
_MSGID_RE = re.compile(r'^Message-ID:\s*(.+?)\s*$', re.MULTILINE | re.IGNORECASE)


def _join_fetch_bytes(lines) -> bytes:
    """Concatenate every bytes-ish line from an aioimaplib fetch response."""
    return b"\n".join(bytes(ln) for ln in lines if isinstance(ln, (bytes, bytearray)))


def _extract_message_id(raw_bytes: bytes) -> str:
    """Pull Message-ID out of raw response bytes (header or full payload)."""
    if not raw_bytes:
        return ""
    text = raw_bytes.decode("utf-8", errors="replace")
    m = _MSGID_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_magic_link(raw_bytes: bytes) -> str | None:
    """Pull the magic-link URL out of an email's text or html body."""
    msg = email.message_from_bytes(raw_bytes)
    bodies: list[str] = []
    for part in msg.walk():
        ct = part.get_content_type()
        if ct not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            bodies.append(payload.decode(charset, errors="replace"))
        except Exception:
            continue
    full = "\n".join(bodies)
    m = MAGIC_LINK_RE.search(full)
    return m.group(0) if m else None


class GmailMagicLinkPlugin(BasePlugin):
    name = "gmail_magic_link"
    version = "2.0.0"
    description = "Forward Claude.ai magic-link emails to Discord (no email deletion, local dedupe)"

    async def on_load(self):
        self._email = os.getenv("GMAIL_ADDRESS", "")
        self._password = os.getenv("GMAIL_APP_PASSWORD", "")
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

        if not self._email or not self._password:
            self.logger.warning(
                "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — plugin idle"
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

    # ── Main loop ────────────────────────────────────────────────────────

    async def _watch_loop(self):
        attempt = 0
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
                attempt = 0  # successful connect resets the backoff

                await self._scan_and_process(imap)

                while not self._stop.is_set():
                    idle_task = await imap.idle_start(timeout=IDLE_TIMEOUT_SECONDS)
                    try:
                        await asyncio.wait_for(
                            imap.wait_server_push(),
                            timeout=IDLE_WAIT_TIMEOUT_SECONDS,
                        )
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
                backoff = RECONNECT_BACKOFF_SCHEDULE[
                    min(attempt, len(RECONNECT_BACKOFF_SCHEDULE) - 1)
                ]
                self.logger.warning(
                    f"IMAP loop error ({type(e).__name__}: {e}); "
                    f"reconnecting in {backoff}s (attempt #{attempt + 1})"
                )
                attempt += 1
                # Close the current connection BEFORE sleeping so the next
                # iteration starts with a clean slate. logout() can itself
                # hang on a half-dead socket, hence the timeout.
                await self._close_imap(imap)
                imap = None
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
            finally:
                if imap is not None:
                    await self._close_imap(imap)

    async def _close_imap(self, imap):
        """Best-effort connection teardown with a timeout.

        Swallows everything except CancelledError so we never wedge the
        reconnect loop on a stuck logout.
        """
        if imap is None:
            return
        try:
            await asyncio.wait_for(imap.logout(), timeout=5)
        except asyncio.CancelledError:
            raise
        except BaseException:
            pass

    # ── Email handling ───────────────────────────────────────────────────

    async def _scan_and_process(self, imap):
        """Search ALL matching messages (any flag state), skip already-processed.

        Treats search failures and a burst of consecutive fetch failures as
        connection-level errors and re-raises — the watch loop will reconnect.
        Isolated per-uid errors are still logged and skipped.
        """
        # uid_search failing is connection-level: re-raise so we reconnect
        # instead of silently entering IDLE on a dead socket (which is how
        # this plugin previously wedged itself).
        resp = await imap.uid_search("SUBJECT", f'"{SUBJECT_FILTER}"')
        if resp.result != "OK":
            raise RuntimeError(f"uid_search failed: {resp}")
        ids_line = resp.lines[0] if resp.lines else b""
        uids = ids_line.decode().split() if ids_line else []
        if not uids:
            return

        # First-run seeding: if this plugin has never processed anything before,
        # treat every existing matching email as already-processed. Only emails
        # that arrive AFTER first startup will be acted on.
        has_history = await asyncio.to_thread(email_dedupe.has_any, PLUGIN_NAME)
        if not has_history:
            seeded = await self._seed_backlog(imap, uids)
            self.logger.info(
                f"First run — seeded {seeded} existing matching email(s) as processed; "
                f"backlog ignored"
            )
            return

        consecutive_failures = 0
        for uid in uids:
            try:
                await self._process_one(imap, uid)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"Failed to process uid={uid}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FETCH_FAILURES:
                    raise RuntimeError(
                        f"{consecutive_failures} consecutive UID-fetch failures "
                        f"(last uid={uid}); assuming IMAP connection is dead"
                    ) from e

    async def _seed_backlog(self, imap, uids) -> int:
        """Fetch Message-IDs (PEEK — no Seen flag) and seed the dedupe store."""
        msg_ids: list[str] = []
        for uid in uids:
            r = await imap.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
            if r.result != "OK":
                continue
            raw = _join_fetch_bytes(r.lines)
            mid = _extract_message_id(raw)
            if mid:
                msg_ids.append(mid)
        return await asyncio.to_thread(
            email_dedupe.bulk_mark_processed, PLUGIN_NAME, msg_ids
        )

    async def _process_one(self, imap, uid: str):
        # Cheap check first: peek just the Message-ID header.
        r = await imap.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
        if r.result != "OK":
            self.logger.warning(f"header-fetch uid={uid} failed: {r}")
            return
        mid = _extract_message_id(_join_fetch_bytes(r.lines))
        if mid and await asyncio.to_thread(email_dedupe.is_processed, PLUGIN_NAME, mid):
            return  # already done

        # Fetch the full body WITHOUT setting \Seen.
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
            mid = _extract_message_id(raw)  # fallback from full payload

        link = _extract_magic_link(raw)
        if not link:
            self.logger.warning(
                f"uid={uid} mid={mid[:60]}: no magic-link URL found in body"
            )
            # Don't mark processed — we may want to retry after a regex fix.
            return

        self.logger.info(
            f"uid={uid} mid={mid[:60]}: forwarding magic link to channel {TARGET_CHANNEL_ID}"
        )
        await self.ctx.send_message(TARGET_CHANNEL_ID, link)

        if mid:
            await asyncio.to_thread(email_dedupe.mark_processed, PLUGIN_NAME, mid)
        else:
            self.logger.warning(
                f"uid={uid}: forwarded but no Message-ID found — "
                f"future scans may re-forward this email"
            )

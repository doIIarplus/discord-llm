"""Watch a Gmail inbox via IMAP IDLE for Claude.ai magic-link emails
and forward the link to a Discord channel, then delete the email."""

import asyncio
import email
import logging
import os
import re

from aioimaplib import aioimaplib

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.gmail_magic_link")

TARGET_CHANNEL_ID = 1483701977954123806
SUBJECT_FILTER = "Secure link to log in to Claude.ai"
MAGIC_LINK_RE = re.compile(r'https://claude\.ai/magic-link(?:\?[^\s"\'<>#]*)?#[^\s"\'<>]+')

IMAP_HOST = "imap.gmail.com"
IDLE_TIMEOUT_SECONDS = 25 * 60  # Gmail kicks IDLE around 29 min — refresh before that
RECONNECT_BACKOFF_INITIAL = 5
RECONNECT_BACKOFF_MAX = 300


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
    version = "1.0.0"
    description = "Forward Claude.ai magic-link emails from Gmail to a Discord channel"

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

                # Catch up on anything that arrived while we were offline.
                await self._scan_and_process(imap)

                # Push loop.
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
                    return  # stop event set
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
        """Find and forward all matching UNSEEN messages, deleting each one."""
        # Gmail returns sequence-number SEARCH unless we ask for UID SEARCH.
        # Build a quoted subject literal.
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

        self.logger.info(f"Found {len(uids)} matching email(s): {uids}")

        for uid in uids:
            try:
                fetch_resp = await imap.uid("fetch", uid, "(RFC822)")
                if fetch_resp.result != "OK":
                    self.logger.warning(f"fetch uid={uid} failed: {fetch_resp}")
                    continue

                raw = self._extract_rfc822(fetch_resp.lines)
                if not raw:
                    self.logger.warning(f"uid={uid}: no RFC822 payload in fetch response")
                    continue

                link = _extract_magic_link(raw)
                if not link:
                    self.logger.warning(f"uid={uid}: no magic link found in body")
                    continue

                self.logger.info(f"uid={uid}: forwarding magic link to channel {TARGET_CHANNEL_ID}")
                await self.ctx.send_message(TARGET_CHANNEL_ID, link)

                # Mark deleted, then expunge so it actually disappears.
                await imap.uid("store", uid, "+FLAGS", "(\\Deleted)")
                await imap.expunge()
                self.logger.info(f"uid={uid}: deleted")
            except Exception as e:
                self.logger.exception(f"Failed to process uid={uid}: {e}")

    @staticmethod
    def _extract_rfc822(lines) -> bytes | None:
        """aioimaplib FETCH returns multiple lines; the body is typically the
        second element. Find the first bytes object that looks like an email."""
        for ln in lines:
            if isinstance(ln, (bytes, bytearray)) and len(ln) > 100:
                return bytes(ln)
        return None

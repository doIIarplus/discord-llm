"""Scheduler plugin — runs due tasks from tools/scheduler/tasks.json every 60 seconds."""

import asyncio
import logging
import os
import subprocess

from plugin_base import BasePlugin, HookType

logger = logging.getLogger("Plugin.scheduler")

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUN_DUE_SCRIPT = os.path.join(PROJECT_DIR, "tools", "scheduler", "run_due.py")
CHECK_INTERVAL = 60  # seconds


class SchedulerPlugin(BasePlugin):
    name = "scheduler"
    version = "1.0.0"
    description = "Runs scheduled tasks automatically (checks every 60s)"

    async def on_load(self):
        self._running = True
        self._task = None
        self.register_hook(HookType.ON_READY, self._on_ready)

    async def on_unload(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _on_ready(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._scheduler_loop())
            logger.info("Scheduler loop started")

    async def _scheduler_loop(self):
        try:
            while self._running:
                await asyncio.sleep(CHECK_INTERVAL)
                await self._run_due_tasks()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}", exc_info=True)

    async def _run_due_tasks(self):
        """Shell out to run_due.py to execute any tasks that are due."""
        if not os.path.exists(RUN_DUE_SCRIPT):
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", RUN_DUE_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=PROJECT_DIR,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=180,
            )

            if proc.returncode == 0:
                import json
                result = json.loads(stdout.decode())
                executed = result.get("executed", [])
                if executed:
                    for task in executed:
                        status = "OK" if task.get("success") else "FAILED"
                        logger.info(f"Scheduled task '{task.get('name')}': {status}")
            else:
                logger.error(f"run_due.py failed (rc={proc.returncode}): {stderr.decode()[:200]}")

        except asyncio.TimeoutError:
            logger.error("run_due.py timed out (180s)")
        except Exception as e:
            logger.error(f"Error running scheduled tasks: {e}")

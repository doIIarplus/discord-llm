"""Background coding jobs, with live progress rendered into a Discord message.

A chat turn must stay fast, but real code work takes minutes. So the chat turn
only *starts* a job; this module runs it detached, streams the agent's progress
into one continuously-edited Discord message, and leaves the result on a branch
until the user says to push it.

Design notes:
  - One active job per channel. Two agents editing the same repo concurrently is
    a mess, and the progress message would fight itself.
  - Discord ratelimits message edits, so updates are coalesced to at most one
    every EDIT_INTERVAL seconds no matter how fast the agent works.
  - Nothing is pushed here. The job ends in `awaiting_push` with a diffstat; the
    user has to ask. That is the one irreversible step, so it stays manual.
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import agent_workspace
from agent_events import ProgressState

logger = logging.getLogger("agent_jobs")

EDIT_INTERVAL = 5.0          # seconds between Discord message edits
MAX_JOB_TURNS = 0            # 0 = no cap; CLAUDE_CODE_TIMEOUT still applies

_TERMINAL = {"done", "failed", "cancelled", "awaiting_push", "pushed"}


@dataclass
class AgentJob:
    id: str
    guild_id: int
    channel_id: int
    repo: str
    task: str
    requester_id: Optional[str] = None
    status: str = "queued"          # queued|running|awaiting_push|done|failed|cancelled
    checkout: str = ""
    worktree: str = ""
    branch: str = ""
    base_ref: str = ""
    session_id: Optional[str] = None
    pushed: bool = False
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    error: Optional[str] = None
    summary: dict = field(default_factory=dict)
    message_id: Optional[int] = None
    _task: Optional[asyncio.Task] = None
    _progress: Optional[ProgressState] = None

    @property
    def elapsed(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    def public(self) -> dict:
        return {
            "job_id": self.id, "status": self.status, "repo": self.repo,
            "task": self.task, "branch": self.branch, "worktree": self.worktree,
            "elapsed_s": round(self.elapsed, 1), "error": self.error,
            "summary": self.summary, "session_id": self.session_id,
        }


class JobManager:
    """Owns running jobs and their lifecycle."""

    def __init__(self, bot, claude_client):
        self.bot = bot
        self.claude = claude_client
        self.jobs: Dict[str, AgentJob] = {}
        self._by_channel: Dict[tuple, str] = {}      # (guild, channel) -> job id

    # ---------- lookup ----------

    def get(self, job_id: str) -> Optional[AgentJob]:
        return self.jobs.get(job_id)

    def active_for(self, guild_id, channel_id) -> Optional[AgentJob]:
        job = self.jobs.get(self._by_channel.get((int(guild_id), int(channel_id)), ""))
        return job if job and job.status not in _TERMINAL else None

    def latest_for(self, guild_id, channel_id) -> Optional[AgentJob]:
        """Most recent job in a channel, terminal or not (for 'push it')."""
        candidates = [j for j in self.jobs.values()
                      if j.guild_id == int(guild_id) and j.channel_id == int(channel_id)]
        return max(candidates, key=lambda j: j.started_at, default=None)

    # ---------- lifecycle ----------

    async def start(self, guild_id, channel_id, repo, task, requester_id=None) -> dict:
        guild_id, channel_id = int(guild_id), int(channel_id)
        existing = self.active_for(guild_id, channel_id)
        if existing:
            raise RuntimeError(
                f"a job is already running in this channel ({existing.id}, "
                f"{existing.status}, {existing.elapsed:.0f}s in). Wait for it or "
                "cancel it first.")

        job = AgentJob(id=uuid.uuid4().hex[:12], guild_id=guild_id,
                       channel_id=channel_id, repo=repo, task=task,
                       requester_id=str(requester_id) if requester_id else None)

        # Worktree creation is quick but can fail (repo not cloned, mid-rebase).
        # Do it synchronously so the caller gets a real error instead of a job
        # that dies a second later.
        wt = await asyncio.to_thread(
            agent_workspace.create, repo, task, job.id)
        job.repo = wt["repo"]
        job.checkout, job.worktree = wt["checkout"], wt["path"]
        job.branch, job.base_ref = wt["branch"], wt["start"]
        job._progress = ProgressState(task=task, repo=job.repo)

        self.jobs[job.id] = job
        self._by_channel[(guild_id, channel_id)] = job.id
        job._task = asyncio.create_task(self._run(job))
        logger.info("[agent] job %s started: %s in %s", job.id, task[:60], job.worktree)
        return job.public()

    async def cancel(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"no such job: {job_id}")
        if job.status in _TERMINAL:
            return job.public()
        job.status = "cancelled"
        if job._task:
            job._task.cancel()
        return job.public()

    async def shutdown(self) -> None:
        """Cancel everything still running (bot is closing)."""
        for job in list(self.jobs.values()):
            if job.status not in _TERMINAL and job._task:
                job._task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(job._task, return_exceptions=True), timeout=10)
                except Exception:
                    pass

    # ---------- the run ----------

    async def _run(self, job: AgentJob) -> None:
        channel = None
        message = None
        last_edit = 0.0

        async def flush(force=False):
            """Push the current progress view to Discord, rate-limited.

            Dropped updates are fine: the view is a full re-render of current
            state, not a delta, and the `finally` block always does a final
            edit — so the last word is never lost.
            """
            nonlocal last_edit
            if message is None:
                return
            now = time.monotonic()
            if not force and now - last_edit < EDIT_INTERVAL:
                return
            last_edit = now
            try:
                await message.edit(content=job._progress.render(job.elapsed, job.status))
            except Exception as e:
                logger.debug("[agent] progress edit failed: %s", e)

        def on_event(event):
            changed = job._progress.apply(event)
            if job._progress.session_id:
                job.session_id = job._progress.session_id
            if changed:
                return flush()
            return None

        try:
            job.status = "running"
            channel = self.bot.get_channel(job.channel_id) or \
                await self.bot.fetch_channel(job.channel_id)
            message = await channel.send(job._progress.render(0.0, "running"))
            job.message_id = message.id
            last_edit = time.monotonic()

            prompt = self._build_prompt(job)
            result = await self.claude.run_agent_task(
                prompt=prompt,
                cwd=job.worktree,
                on_event=on_event,
                extra_dirs=[agent_workspace.PROJECT_DIR],
                max_turns=MAX_JOB_TURNS,
            )
            job.session_id = result.get("session_id") or job.session_id

            # Commit anything the agent left in the working tree, so the work
            # survives on the branch whether or not it gets pushed.
            await asyncio.to_thread(
                agent_workspace.commit_all, job.worktree,
                f"{job.task[:70]}\n\nvia jaspt agent job {job.id}")
            job.summary = await asyncio.to_thread(
                agent_workspace.summarize, job.worktree, job.base_ref)

            if result.get("is_error"):
                job.status = "failed"
                job.error = (result.get("result") or result.get("stderr") or "")[:500]
            elif not job.summary.get("commits"):
                # Ran fine but changed nothing — a question, or a no-op.
                job.status = "done"
            else:
                # Publish the branch automatically. This is safe: it's a task
                # branch, never the default branch, so nothing the user depends
                # on moves. It also means the result is a real GitHub link
                # instead of something stuck on this machine.
                try:
                    pushed = await asyncio.to_thread(
                        agent_workspace.push_branch, job.worktree, job.branch)
                    job.summary["head_sha"] = pushed["sha"]
                    job.pushed = True
                    job.status = "pushed"
                except agent_workspace.PushDenied as e:
                    # No write access — the work is committed locally and safe.
                    job.status = "awaiting_push"
                    job.error = str(e)
                except Exception as e:
                    job.status = "awaiting_push"
                    job.error = f"push failed: {e}"
                    logger.warning("[agent] job %s push failed: %s", job.id, e)

        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as e:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            logger.exception("[agent] job %s failed", job.id)
        finally:
            job.ended_at = time.time()
            keep_branch = bool((job.summary or {}).get("commits"))
            # Once the branch is pushed the worktree has served its purpose;
            # keep it only when the work is still local-only and might need a
            # retry of the push.
            if job.status != "awaiting_push":
                try:
                    await asyncio.to_thread(
                        agent_workspace.remove, job.checkout, job.worktree,
                        job.branch, keep_branch)
                except Exception as e:
                    logger.warning("[agent] worktree cleanup failed: %s", e)

            final_text = self._final_text(job)
            view = self._result_view(job)
            try:
                if message is not None:
                    await message.edit(content=final_text, view=view)
                elif channel is not None:
                    # Failed before the progress message existed — still report.
                    await channel.send(final_text, view=view)
            except Exception as e:
                logger.debug("[agent] final report failed: %s", e)
            logger.info("[agent] job %s -> %s in %.0fs", job.id, job.status, job.elapsed)

    def _build_prompt(self, job: AgentJob) -> str:
        return (
            f"You are working in a git worktree of {job.repo}, on branch "
            f"{job.branch}, branched from {job.base_ref}. The working directory "
            f"is already this worktree — treat it as the whole project.\n\n"
            f"TASK: {job.task}\n\n"
            "Do the work properly: read the surrounding code first and match its "
            "conventions. Use subagents (the Task tool) when exploring several "
            "areas at once. Run the project's tests if it has any, and say so if "
            "they fail.\n"
            "Do NOT commit — the harness commits for you. Do NOT push, and do NOT "
            "run git commands that rewrite history.\n"
            "Finish with a short plain-language summary of what you changed and "
            "why, plus anything you could not do."
        )

    def _result_view(self, job: AgentJob):
        """Buttons for a job that actually produced a pushed branch."""
        if job.status != "pushed":
            return None
        try:
            from agent_views import JobResultView
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
            return JobResultView(job, requester_id=job.requester_id,
                                 github_token=token)
        except Exception as e:
            logger.warning("[agent] could not build result view: %s", e)
            return None

    def _final_text(self, job: AgentJob) -> str:
        body = job._progress.render(job.elapsed, job.status) if job._progress else ""
        s = job.summary or {}
        extra = []

        if job.status == "pushed":
            if s.get("diffstat"):
                extra.append(f"```\n{s['diffstat'][:600]}\n```")
            sha = s.get("head_sha", "")
            if sha:
                extra.append(
                    f"pushed to `{job.branch}` · "
                    f"<https://github.com/{job.repo}/commit/{sha}>")
            else:
                extra.append(f"pushed to `{job.branch}`")
        elif job.status == "awaiting_push":
            if s.get("diffstat"):
                extra.append(f"```\n{s['diffstat'][:600]}\n```")
            extra.append(f"committed to `{job.branch}` but NOT pushed — "
                         f"{job.error or 'push failed'}")
        elif job.status == "failed" and job.error:
            extra.append(f"error: {job.error[:300]}")
        elif job.status == "done":
            extra.append("_(no file changes)_")
        return (body + ("\n" + "\n".join(extra) if extra else ""))[:1990]

    # ---------- push gate ----------

    async def push(self, job_id: str) -> dict:
        """Retry publishing a branch whose automatic push didn't land.

        Jobs push themselves now, so this is the recovery path — e.g. the first
        attempt failed on a transient network error. A job that's already
        `pushed` has nothing to do.
        """
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"no such job: {job_id}")
        if job.status == "pushed":
            return {**job.public(), "pushed_branch": job.branch,
                    "note": "already pushed"}
        if job.status != "awaiting_push":
            raise RuntimeError(f"job {job_id} is {job.status}, nothing to push")

        pushed = await asyncio.to_thread(
            agent_workspace.push_branch, job.worktree, job.branch)
        job.summary["head_sha"] = pushed["sha"]
        job.pushed = True
        job.status = "pushed"
        try:
            await asyncio.to_thread(
                agent_workspace.remove, job.checkout, job.worktree, job.branch, True)
        except Exception as e:
            logger.warning("[agent] post-push cleanup failed: %s", e)
        return {**job.public(), "pushed_branch": job.branch}

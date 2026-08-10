"""Discord buttons for a finished agent job.

The job auto-commits and pushes its branch, so by the time this view appears the
work is already on GitHub — nothing here is required. The buttons are the
*next* steps: open a PR, look at the diff, or throw the branch away.

Why a compare link rather than always creating the PR through the API: PR
creation needs a GitHub token, and there isn't one configured by default. A link
button pointed at GitHub's compare page opens the PR form already filled in,
works with zero credentials, and can't fail. When GITHUB_TOKEN *is* set we do it
properly and open the PR from Discord.

Only the person who asked for the job can press the action buttons — same rule
as RestartConfirmView in commands.py.
"""

import logging

import discord

logger = logging.getLogger("agent_views")

# Long but finite: Discord views don't survive a bot restart, and a view that
# silently stops working is worse than one that says so.
VIEW_TIMEOUT = 24 * 3600


def compare_url(repo: str, base: str, branch: str) -> str:
    """GitHub's prefilled 'open a pull request' page for a branch."""
    return f"https://github.com/{repo}/compare/{base}...{branch}?expand=1"


def commit_url(repo: str, sha: str) -> str:
    return f"https://github.com/{repo}/commit/{sha}"


def branch_url(repo: str, branch: str) -> str:
    return f"https://github.com/{repo}/tree/{branch}"


class JobResultView(discord.ui.View):
    """Post-job actions: open a PR, view the code, delete the branch."""

    def __init__(self, job, requester_id=None, github_token: str = ""):
        super().__init__(timeout=VIEW_TIMEOUT)
        self.job = job
        self.requester_id = int(requester_id) if requester_id else None
        self.github_token = github_token
        self._done = False

        repo, branch, base = job.repo, job.branch, job.base_ref.split("/")[-1]

        # Link buttons first — they always work, no token, no failure mode.
        if self.github_token:
            # Real PR creation is available, so the primary action is a button.
            self.add_item(_OpenPRButton())
        else:
            self.add_item(discord.ui.Button(
                label="Open PR", style=discord.ButtonStyle.link,
                url=compare_url(repo, base, branch), emoji="🔀"))

        sha = (job.summary or {}).get("head_sha")
        self.add_item(discord.ui.Button(
            label="View diff", style=discord.ButtonStyle.link,
            url=commit_url(repo, sha) if sha else branch_url(repo, branch),
            emoji="🔍"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Only the requester may act. Anyone can still click the link buttons."""
        if self.requester_id is None or interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "only whoever asked for this job can use these buttons", ephemeral=True)
        return False

    async def on_timeout(self):
        for item in self.children:
            if not isinstance(item, discord.ui.Button) or item.style != discord.ButtonStyle.link:
                item.disabled = True

    @discord.ui.button(label="Delete branch", style=discord.ButtonStyle.red, emoji="🗑️")
    async def delete_branch(self, interaction: discord.Interaction,
                            button: discord.ui.Button):
        import agent_workspace

        await interaction.response.defer()
        try:
            import asyncio
            await asyncio.to_thread(
                agent_workspace.delete_branch, self.job.checkout,
                self.job.branch, True)
        except Exception as e:
            await interaction.followup.send(f"couldn't delete it: {e}", ephemeral=True)
            return
        for item in self.children:
            item.disabled = True
        self.stop()
        await interaction.followup.send(
            f"deleted `{self.job.branch}` (local + remote)")


class _OpenPRButton(discord.ui.Button):
    """Creates the PR through the API — only added when a token exists."""

    def __init__(self):
        super().__init__(label="Open PR", style=discord.ButtonStyle.green, emoji="🔀")

    async def callback(self, interaction: discord.Interaction):
        view: JobResultView = self.view
        job = view.job
        await interaction.response.defer()

        import asyncio
        import json
        import urllib.error
        import urllib.request

        base = job.base_ref.split("/")[-1]
        title = (job.task or "agent changes").strip().splitlines()[0][:70]
        body = ("Opened by jaspt from Discord.\n\n"
                f"**Task:** {job.task}\n\n"
                f"```\n{(job.summary or {}).get('diffstat', '')[:800]}\n```")

        def _create():
            req = urllib.request.Request(
                f"https://api.github.com/repos/{job.repo}/pulls",
                data=json.dumps({"title": title, "head": job.branch,
                                 "base": base, "body": body}).encode(),
                headers={"Accept": "application/vnd.github+json",
                         "Authorization": f"Bearer {view.github_token}",
                         "User-Agent": "jaspt-agent",
                         "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())

        try:
            pr = await asyncio.to_thread(_create)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode()).get("message", "")
            except Exception:
                pass
            # Fall back to the link that always works rather than dead-ending.
            await interaction.followup.send(
                f"couldn't open the PR via the API ({e.code}: {detail}). "
                f"You can still do it here: {compare_url(job.repo, base, job.branch)}")
            return
        except Exception as e:
            await interaction.followup.send(
                f"PR creation failed: {e}. Manual link: "
                f"{compare_url(job.repo, base, job.branch)}")
            return

        self.disabled = True
        self.label = "PR opened"
        await interaction.followup.send(f"opened PR: {pr.get('html_url')}")

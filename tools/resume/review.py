#!/usr/bin/env python3
"""Score a resume PDF against an optional role description using the local LLM.

Modeled on interviewstreet/hiring-agent: a resume-to-score pipeline that
extracts structured data from a resume PDF, optionally enriches it with public
GitHub signals, and produces an explainable per-category evaluation.

Pipeline:
  1. Extract text from the resume PDF (via pypdf, same approach as file_parser).
  2. Optionally fetch public GitHub profile + repo signals (no auth needed).
  3. Ask the local Ollama chat model to score the candidate across a fixed
     rubric (technical skills, experience, project quality, education,
     overall fit) and explain each score.

Output JSON:
  {
    "overall_score": <int 0-100>,
    "categories": [{"name": ..., "score": <int 0-100>, "reasoning": ...}, ...],
    "summary": "..."
  }

Read-only: only reads the resume file and public web data. No confirmation
needed.

Usage:
  python tools/resume/review.py --pdf resume.pdf
  python tools/resume/review.py --pdf resume.pdf --role-description "Senior backend engineer..."
  python tools/resume/review.py --pdf resume.pdf --role-file jd.txt --github-username octocat
"""

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Add project root to path so we can import config, ollama_client, sandbox.
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

from sandbox import safe_path, SandboxViolation
import config
from ollama_client import OllamaClient

# Fixed rubric mirroring hiring-agent's evaluation categories.
RUBRIC = [
    ("technical_skills", "Technical skills — depth and breadth of relevant technologies, languages, and tools."),
    ("experience", "Experience — seniority, scope, and relevance of past roles and responsibilities."),
    ("project_quality", "Project quality — impact, complexity, and outcomes of projects and contributions."),
    ("education", "Education — relevant degrees, certifications, and continuous learning."),
    ("overall_fit", "Overall fit — how well the candidate matches the target role and team needs."),
]

GITHUB_API = "https://api.github.com"


def read_pdf_text(pdf_path: str) -> str:
    """Extract text from a resume PDF using pypdf (mirrors file_parser._parse_pdf)."""
    try:
        import pypdf
    except ImportError:
        error("pypdf not installed. Run: pip install pypdf")

    try:
        path = safe_path(pdf_path)
    except SandboxViolation as e:
        error(f"Path not allowed: {e}")

    if not os.path.exists(path):
        error(f"Resume PDF not found: {pdf_path}")
    if os.path.splitext(path)[1].lower() != ".pdf":
        error(f"Not a PDF file: {pdf_path}")

    text_content = []
    try:
        reader = pypdf.PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    except Exception as e:
        error(f"PDF parsing failed: {e}")

    text = "\n".join(text_content).strip()
    if not text:
        error("No extractable text found in the PDF (it may be scanned/image-only).")
    return text


def _gh_get(path: str) -> object:
    """GET a public GitHub REST endpoint. Returns parsed JSON or None on error."""
    req = urllib.request.Request(
        f"{GITHUB_API}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "discord-llm-bot-resume-review",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, OSError):
        return None


def fetch_github_signals(username: str) -> dict:
    """Fetch public GitHub profile + top-repo signals. Best-effort; returns
    a dict with an 'error' key if the profile can't be fetched."""
    profile = _gh_get(f"/users/{username}")
    if not profile or "login" not in profile:
        return {"error": f"Could not fetch public GitHub profile for '{username}'."}

    repos = _gh_get(f"/users/{username}/repos?per_page=100&sort=updated&type=owner") or []
    # Rank owned, non-fork repos by stars.
    owned = [r for r in repos if isinstance(r, dict) and not r.get("fork")]
    owned.sort(key=lambda r: r.get("stargazers_count", 0), reverse=True)

    languages = {}
    total_stars = 0
    for r in owned:
        total_stars += r.get("stargazers_count", 0)
        lang = r.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1

    top_repos = []
    for r in owned[:8]:
        top_repos.append({
            "name": r.get("name"),
            "description": r.get("description"),
            "language": r.get("language"),
            "stars": r.get("stargazers_count", 0),
            "forks": r.get("forks_count", 0),
        })

    top_langs = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "login": profile.get("login"),
        "name": profile.get("name"),
        "bio": profile.get("bio"),
        "public_repos": profile.get("public_repos"),
        "followers": profile.get("followers"),
        "total_stars_owned": total_stars,
        "top_languages": [lang for lang, _ in top_langs[:8]],
        "top_repos": top_repos,
    }


def build_prompt(resume_text: str, role_text: str, github: dict) -> str:
    """Build the scoring prompt for the LLM."""
    rubric_lines = "\n".join(f"  - {key}: {desc}" for key, desc in RUBRIC)
    categories_json = ", ".join(f'"{key}"' for key, _ in RUBRIC)

    role_block = (
        f"TARGET ROLE / JOB DESCRIPTION:\n{role_text}\n\n"
        if role_text
        else "TARGET ROLE / JOB DESCRIPTION:\n(none provided — score for a general "
             "software engineering role.)\n\n"
    )

    github_block = ""
    if github:
        if github.get("error"):
            github_block = f"GITHUB SIGNALS:\n{github['error']}\n\n"
        else:
            github_block = "GITHUB SIGNALS (public profile):\n" + json.dumps(github, indent=2) + "\n\n"

    system = (
        "You are a hiring evaluation assistant. Given a candidate's resume, an "
        "optional target role, and optional GitHub signals, you produce an "
        "explainable, evidence-based evaluation across a fixed rubric.\n\n"
        "Score each category from 0 to 100 and give a short, specific reason that "
        "cites concrete evidence from the resume/GitHub. Then give an overall "
        "score from 0 to 100 and a brief summary.\n\n"
        "Be objective and grounded — do not invent qualifications the candidate "
        "does not clearly have. If evidence for a category is thin, score it lower "
        "and say why.\n\n"
        f"RUBRIC (score each of these categories):\n{rubric_lines}\n\n"
        "OUTPUT FORMAT — respond with ONLY a single JSON object, no markdown, no "
        "code fences, no commentary. Schema:\n"
        "{\n"
        '  "overall_score": <integer 0-100>,\n'
        '  "categories": [\n'
        f'    {{"name": <one of: {categories_json}>, "score": <integer 0-100>, "reasoning": <string>}},\n'
        "    ... one entry per rubric category ...\n"
        "  ],\n"
        '  "summary": <string, 1-3 sentences>\n'
        "}"
    )

    return (
        f"System: {system}\n\n"
        f"{role_block}"
        f"{github_block}"
        f"RESUME TEXT:\n{resume_text}\n\n"
        f"Assistant (JSON only):"
    )


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of the model's response."""
    text = raw.strip()
    # Strip code fences if present.
    fence = re.match(r"^```[a-zA-Z]*\n?(.+?)\n?```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Fallback: grab the outermost {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    error("Model did not return valid JSON.", details=raw[:2000])


def _clamp_score(v) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def normalize_result(parsed: dict) -> dict:
    """Coerce the model output into the documented schema."""
    valid_names = {key for key, _ in RUBRIC}
    categories = []
    for cat in parsed.get("categories", []) or []:
        if not isinstance(cat, dict):
            continue
        name = str(cat.get("name", "")).strip()
        categories.append({
            "name": name,
            "score": _clamp_score(cat.get("score")),
            "reasoning": str(cat.get("reasoning", "")).strip(),
        })

    overall = parsed.get("overall_score")
    if overall is None and categories:
        overall = round(sum(c["score"] for c in categories) / len(categories))
    return {
        "overall_score": _clamp_score(overall),
        "categories": categories,
        "summary": str(parsed.get("summary", "")).strip(),
    }


async def _run(prompt: str) -> str:
    client = OllamaClient()
    return await client.generate(
        prompt,
        model=config.CHAT_MODEL,
        keep_alive=300,
        num_ctx=16384,
        num_predict=1500,
        think=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf", required=True, help="Path to the resume PDF (within project dir)")
    parser.add_argument("--role-description", help="Job description text to score against")
    parser.add_argument("--role-file", help="Path to a file containing the job description")
    parser.add_argument("--github-username", help="Public GitHub username to enrich signals with")
    args = parser.parse_args()

    resume_text = read_pdf_text(args.pdf)

    role_text = ""
    if args.role_description:
        role_text = args.role_description.strip()
    elif args.role_file:
        try:
            with open(safe_path(args.role_file), "r", encoding="utf-8", errors="replace") as f:
                role_text = f.read().strip()
        except SandboxViolation as e:
            error(f"Path not allowed: {e}")
        except OSError as e:
            error(f"Could not read role file: {e}")

    github = {}
    if args.github_username:
        github = fetch_github_signals(args.github_username.strip())

    prompt = build_prompt(resume_text, role_text, github)

    try:
        raw = asyncio.run(_run(prompt))
    except Exception as e:
        error(f"LLM query failed: {type(e).__name__}: {e}")

    parsed = _extract_json(raw)
    result = normalize_result(parsed)
    output(result)


if __name__ == "__main__":
    main()

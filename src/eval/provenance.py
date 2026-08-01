"""Git provenance for logged experiment runs (CLAUDE.md code standards:
"Every experiment logs: git commit SHA, random seed, full config, timestamp").

A bare `git rev-parse HEAD` is not sufficient: it reports the last commit
regardless of uncommitted changes, so a run against a dirty working tree logs
a SHA that does not correspond to the code that actually produced its numbers.
Every experiment run in this project so far predates this fix and was run
against a dirty tree, so their logged SHA identifies the parent commit their
changes were based on, not the exact code -- documented explicitly rather than
silently treated as exact provenance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_sha(repo_root: Path) -> str:
    """HEAD's commit SHA, suffixed "-dirty" if the working tree has changes
    relative to it. "unknown" if this is not a git checkout at all.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"

    dirty = (
        subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=repo_root,
        ).returncode
        != 0
    )
    return f"{sha}-dirty" if dirty else sha

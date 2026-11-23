"""Link every result to the exact code that produced it.

Each run folder is named after the Git commit of the code that ran it:
runs/<experiment>/<commit>/. A run refuses to start while the repo has
uncommitted changes, because those results would point to a commit that
lacks the code that made them. Pass --allow-dirty for quick experiments;
the folder name then ends in "-dirty" as a warning label.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(base: str, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", base, *args], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(base: str) -> dict:
    commit = _git(base, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "short": "nogit", "dirty": True, "branch": None, "tag": None}
    status = _git(base, "status", "--porcelain") or ""
    return {
        "commit": commit,
        "short": commit[:8],
        "dirty": bool(status),
        "changed_files": status.splitlines()[:20],
        "branch": _git(base, "rev-parse", "--abbrev-ref", "HEAD"),
        "tag": _git(base, "describe", "--tags", "--exact-match"),
    }


class DirtyRepoError(RuntimeError):
    pass


def run_dir_for(cfg: dict) -> tuple[Path, dict]:
    """Return the run folder for this config and the code-version record."""
    info = git_info(cfg["_base"])
    if info["dirty"] and not cfg.get("_allow_dirty"):
        if info["commit"] is None:
            msg = "This folder is outside a Git repository, so results could not be linked to a code version."
        else:
            msg = "The repo has uncommitted changes:\n  " + "\n  ".join(info["changed_files"])
        raise DirtyRepoError(msg + "\nCommit your changes first, or rerun with --allow-dirty for a throwaway run.")
    label = info["short"] + ("-dirty" if info["dirty"] and info["commit"] else "")
    info["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Path(cfg["_base"]) / cfg["output_dir"] / cfg["experiment"]["name"] / label, info

"""Reject local-only project material from the public Git repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


FORBIDDEN_NAMES = {"agents.md", "claude.md", "handoff.md"}
FORBIDDEN_PARTS = {"non-sharable", "non_sharable", "nonsharable"}


def classify_paths(paths: Iterable[str]) -> List[Tuple[str, str]]:
    """Return ``(path, reason)`` pairs for paths outside the share boundary."""
    violations: List[Tuple[str, str]] = []
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/").strip("/")
        if not normalized:
            continue
        parts = [part.casefold() for part in normalized.split("/")]
        if parts[-1] in FORBIDDEN_NAMES:
            violations.append((raw_path, "local agent/handoff document"))
        elif any(part in FORBIDDEN_PARTS for part in parts):
            violations.append((raw_path, "Non-Sharable content"))
    return violations


def _git_paths(repo: Path, *args: str, nul: bool = False) -> List[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    separator = b"\0" if nul else b"\n"
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in completed.stdout.split(separator)
        if value
    ]


def audit_repository(repo: Path) -> List[Tuple[str, str]]:
    violations = classify_paths(_git_paths(repo, "ls-files", "-z", nul=True))
    history_paths = []
    for record in _git_paths(repo, "rev-list", "--objects", "--all"):
        _, separator, path = record.partition(" ")
        if separator:
            history_paths.append(path)
    violations.extend(classify_paths(history_paths))
    return sorted(set(violations))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    try:
        violations = audit_repository(repo)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Share-boundary audit could not run: {exc}", file=sys.stderr)
        return 2
    if violations:
        print("Non-shareable paths were found in Git:", file=sys.stderr)
        for path, reason in violations:
            print(f"  - {path}: {reason}", file=sys.stderr)
        return 1
    print("Share boundary OK: no local agent or Non-Sharable paths are tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

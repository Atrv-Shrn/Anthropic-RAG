"""Load the curated Anthropic repo list from ``config/repos.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml

REPOS_YAML = Path(__file__).with_name("repos.yaml")


class RepoSpec(NamedTuple):
    org: str
    name: str

    @property
    def full(self) -> str:
        return f"{self.org}/{self.name}"


def load_repos(path: Path | str = REPOS_YAML) -> tuple[str, list[str]]:
    """Return ``(org, [repo_names])`` from the YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    org = data.get("org", "anthropics")
    repos = [str(r) for r in data.get("repos", [])]
    return org, repos
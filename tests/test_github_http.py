"""Unit tests for the GitHub REST helper: Link parsing, resolve_repo, paginate."""

from __future__ import annotations


from config.settings import Settings
from rag.corpus import _github_http as gh


def _settings() -> Settings:
    # A token is optional for these tests; pass an explicit one so _headers is stable.
    return Settings(github_token="testtoken")


def test_next_link_extracts_rel_next():
    header = (
        '<https://api.github.com/x?page=2>; rel="next", '
        '<https://api.github.com/x?page=9>; rel="last"'
    )
    assert gh._next_link(header) == "https://api.github.com/x?page=2"


def test_next_link_absent_returns_none():
    assert gh._next_link('<https://api.github.com/x?page=9>; rel="last"') is None
    assert gh._next_link("") is None


def test_headers_include_bearer_when_token_set():
    h = gh._headers(_settings())
    assert h["Authorization"] == "Bearer testtoken"
    assert h["Accept"] == "application/vnd.github+json"


def test_resolve_repo_follows_move_and_reads_default_branch(monkeypatch):
    # Simulate the dxt -> modelcontextprotocol/mcpb transfer with a master default.
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "owner": {"login": "modelcontextprotocol"},
                "name": "mcpb",
                "default_branch": "main",
            }

    class _Client:
        def __init__(self, *a, **k):
            captured["follow_redirects"] = k.get("follow_redirects")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            captured["url"] = url
            return _Resp()

    monkeypatch.setattr(gh.httpx, "Client", _Client)
    info = gh.resolve_repo("anthropics", "dxt", settings=_settings())
    assert info.owner == "modelcontextprotocol"
    assert info.name == "mcpb"
    assert info.default_branch == "main"
    assert captured["follow_redirects"] is True  # must follow the 301


def test_resolve_repo_defaults_branch_to_main_when_missing(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"owner": {"login": "anthropics"}, "name": "skills"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _Resp()

    monkeypatch.setattr(gh.httpx, "Client", _Client)
    info = gh.resolve_repo("anthropics", "skills", settings=_settings())
    assert info.default_branch == "main"


def test_paginate_follows_link_until_exhausted(monkeypatch):
    pages = [
        (["a", "b"], '<https://api.github.com/next1>; rel="next"'),
        (["c"], '<https://api.github.com/next2>; rel="next"'),
        ([], ""),
    ]
    calls = {"i": 0}

    class _Resp:
        def __init__(self, data, link):
            self._data = data
            self.headers = {"link": link}

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            data, link = pages[calls["i"]]
            calls["i"] += 1
            return _Resp(data, link)

    monkeypatch.setattr(gh.httpx, "Client", _Client)
    out = list(gh.paginate("/repos/x/y/issues", settings=_settings()))
    assert out == ["a", "b", "c"]

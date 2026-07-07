# Skill: anthropic-rag

Grounded Q&A over the **Anthropic GitHub organization** — its documentation, issues,
pull requests, and comments (not source code). The corpus is kept fresh by hourly
incremental ingestion. Use it whenever you need an answer that is *cited to real
Anthropic GitHub content* instead of your own recollection.

## When to use it

- A question about Anthropic's SDKs, docs, or GitHub activity where you want a grounded,
  source-backed answer rather than a guess.
- You want to scope a question to one kind of content (only PRs, only issues, only
  comments, only docs).
- You want to browse or pull the **raw** source documents yourself (e.g. read the full
  text of a PR or doc), not just a synthesized answer.

Don't use it for questions unrelated to the Anthropic GitHub org — it will refuse or
return nothing useful.

## Tools

- **`answer(query)`** — grounded answer over everything (docs + issues + PRs + comments).
  Returns `{answer, source_doc_ids}`.
- **`answer_prs(query)`** — same, but grounded only in pull requests.
- **`answer_issues(query)`** — grounded only in issues.
- **`answer_comments(query)`** — grounded only in comments (issue + PR review comments).
- **`answer_docs(query)`** — grounded only in documentation files (`.md`/`.rst`/`.ipynb`).
- **`list_documents(category=None, repo=None, limit=50, offset=0)`** — browse the raw
  docstore. `category` is one of `prs`/`issues`/`comments`/`docs` (or omit for all);
  `repo` is a full `org/repo` (e.g. `anthropics/anthropic-sdk-python`). Returns
  `{total, limit, offset, documents:[{doc_id, metadata, text_preview}]}`.
- **`get_documents(doc_ids)`** — fetch the full raw text + metadata for specific IDs.
  IDs come from `answer*` (`source_doc_ids`) or `list_documents` (`doc_id`).

## How to use it

- **Just want an answer:** call `answer` (or a category variant to narrow it). The
  `source_doc_ids` tell you exactly what it was grounded on — pass any of them to
  `get_documents` to read the underlying source.
- **Want to read raw sources yourself:** `list_documents(category=..., repo=...)` to find
  candidates, then `get_documents([...])` on the `doc_id`s you care about. Page with
  `limit`/`offset` when `total` exceeds one page.

### Example

> "Were there any recent PRs about streaming in the Python SDK?"

1. `answer_prs("recent changes to streaming in the Python SDK")` → grounded summary +
   `source_doc_ids`.
2. `get_documents(source_doc_ids)` → full PR text if you need the details.

Or, to browse first: `list_documents(category="prs", repo="anthropics/anthropic-sdk-python", limit=10)`
→ pick `doc_id`s → `get_documents([...])`.

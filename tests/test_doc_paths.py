"""Every `app/...py` path mentioned in docs and comments must actually exist.

This is the check the 08-22 review argued about and concluded was worth
having. One reviewer held that a doc-path check would catch nothing,
because the drift it had found was semantic -- prose describing behaviour
the code no longer had. Another then found seven dead module paths in
`.env.example` alone, which a check like this catches trivially. Both were
right about different things, so this covers the mechanical half and
leaves the semantic half to review.

The paths went stale in one specific way worth knowing: `ebf0b35`
reorganised `app/` into packages (`api/`, `db/`, `ingestion/`, `llm/`,
`retrieval/`), and the follow-up doc sweeps fixed README, CLAUDE.md,
AGENTS.md and the source docstrings while missing `.env.example` and two
Cloud Run YAMLs entirely. Three commits in a row fixed instances of this
class and declared it closed; this test is what makes "closed" checkable
instead of asserted.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Text files a stale path can hide in. Deliberately includes .py: module
# docstrings and comments cross-reference sibling modules constantly, and
# those references rot exactly like the ones in Markdown do.
SCANNED_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".html"}
# Extensionless files that still carry instructions a reader follows.
# Dockerfile earns its place the hard way: it told operators to run
# `python -m app.ingest` long after that module moved, and no sweep saw it
# because every scan keyed on a suffix this file does not have.
SCANNED_NAMES = {".env.example", "Dockerfile"}

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", ".fastembed_cache", "htmlcov",
    # notes/ is gitignored working material, and most of it is dated
    # changelogs -- "phase 6 moved app/circuit.py" is an accurate record of
    # what the tree looked like then. Rewriting those paths would falsify
    # the history rather than fix a defect, so they are out of scope.
    "notes",
}

# This file necessarily contains the very paths it exists to reject, as
# examples and as the parametrised premise check below.
SKIP_FILES = {"test_doc_paths.py"}

# Matches app/foo.py and app/sub/foo.py, the form used throughout this repo.
# Dotted module paths (app.config) are deliberately not matched: they appear
# in prose far more loosely and would produce false positives.
PATH_RE = re.compile(r"\bapp/[A-Za-z0-9_/]+\.py\b")


def _scanned_files() -> list[Path]:
    out = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in SCANNED_SUFFIXES or path.name in SCANNED_NAMES:
            out.append(path)
    return out


def test_scan_actually_finds_files():
    """Guard against the scan silently matching nothing and passing."""
    files = _scanned_files()
    assert len(files) > 20, f"expected to scan the repo, only found {len(files)} files"


def test_referenced_module_paths_exist():
    dead: list[str] = []

    for file in _scanned_files():
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in set(PATH_RE.findall(text)):
            if not (REPO_ROOT / match).is_file():
                rel = file.relative_to(REPO_ROOT).as_posix()
                dead.append(f"{rel}: {match}")

    assert not dead, (
        "Documentation references module paths that do not exist:\n  "
        + "\n  ".join(sorted(dead))
        + "\n\napp/ is organised into packages (api/, db/, ingestion/, llm/, "
          "retrieval/) -- a path like app/memory.py is pre-ebf0b35."
    )


MODULE_RE = re.compile(r"-m\s+(app(?:\.[A-Za-z0-9_]+)+)")


def test_documented_module_invocations_resolve():
    """`python -m app.foo.bar` in docs must name a real module.

    The slash-form check above does not see these: the Dockerfile told
    operators to run `python -m app.ingest` for eight months after
    ebf0b35 moved it to app.ingestion.ingest, and every doc-path sweep
    missed it because the string contains no `/`. Anyone following the
    image's own instruction got ModuleNotFoundError and an unbuilt corpus.
    """
    dead: list[str] = []

    for file in _scanned_files():
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for dotted in set(MODULE_RE.findall(text)):
            rel = Path(dotted.replace(".", "/"))
            if (REPO_ROOT / rel).with_suffix(".py").is_file():
                continue
            if (REPO_ROOT / rel / "__init__.py").is_file():
                continue
            dead.append(f"{file.relative_to(REPO_ROOT).as_posix()}: python -m {dotted}")

    assert not dead, (
        "Documentation tells the reader to run modules that do not exist:\n  "
        + "\n  ".join(sorted(dead))
    )


@pytest.mark.parametrize("stale", ["app/memory.py", "app/middleware.py", "app/database.py"])
def test_known_stale_paths_are_really_gone(stale):
    """Pins the premise of the test above: these are the pre-reorganisation
    names, and if one ever exists again the check silently stops meaning
    anything."""
    assert not (REPO_ROOT / stale).exists()

# Contributing to Iter

## Development install

Iter depends on **UnifiedAiClient**, a sibling repository. In development it is installed
editable from `../UnifiedAiClient`, so clone the two side by side:

```
<any parent directory>/
├── Iter/
└── UnifiedAiClient/
```

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
source .venv/bin/activate             # Linux / macOS

pip install -r requirements_dev.txt   # editable: ../UnifiedAiClient + .
```

`requirements_dev.txt` installs the sibling repository editable, so changes to the AI layer take
effect without reinstalling. `requirements_prod.txt` pins a Git tag instead; bump that tag
whenever a `call_ai`, `get_embedding`, `configure_provider` or `warm_up` signature changes.

### Why there is no `gui` extra

`streamlit` is a hard dependency, not an optional extra. The GUI is a primary interface here and
nobody installs Iter without it, so an extra would cost an install step and buy nothing. This is
a deliberate, one-off decision, recorded in `CLAUDE.md`. Please do not "tidy" it into an extra.

## Running from the project folder

Everything in Iter resolves against the **current working directory**: `config.json`,
`Presets/`, and the `secrets.json` that UnifiedAiClient looks for. Run `iter` from the project or
workspace folder, or nothing will find its configuration. `ARCHITECTURE.md` explains why the
resolution works this way rather than being package-relative.

## Tests

Standard library `unittest` only. No pytest, and nothing to add to `requirements_dev.txt`.

```bash
python -m unittest discover -s tests
```

Every AI call is stubbed, so the suite needs no network access and no API keys. Keep it that way:
a test that reaches `call_ai` is a broken test.

Test modules live directly under `tests/` and are named `test_*.py`, so default discovery finds
them. `tests/test_processor.py` selects the `silent` logger backend for its module, which is what
that backend exists for.

## Code conventions

- `from __future__ import annotations` at the top of every module, and modern type hints
  (`list[str]`, `X | None`). Type hints are mandatory on every signature.
- Google-style docstrings with `Args:` and `Returns:`.
- English for every symbol, comment, and user-facing string.
- No raw `print()` for user-facing output: use `itertool/logger.py`. In particular, nothing
  reachable from `iter mcp` may write to stdout, which carries the JSON-RPC framing.
- Behaviour changes belong in `processor.py`, never in an interface layer.
- Adding a `Preset` field means touching several files by hand. The checklist is in `CLAUDE.md`.

## Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) 1.1.0. Add your
change under `[Unreleased]` in the appropriate group (`Added`, `Changed`, `Fixed`, `Removed`),
written for someone reading the project from outside rather than as a commit message.

Versioning and tagging are done by the maintainer.

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Runs are resumable.** Every completed AI call is saved at once to a progress file
  (`<output>.iter-progress.jsonl`, or a file in the working directory when there is no
  output), so a run that stops early resumes where it left off. The file is deleted when the
  run completes. If the input, the preset or the model settings changed in between, Iter asks
  whether to continue, redoing only the nodes whose content changed, or to restart. The CLI
  asks in the console, the GUI in a dialog. The MCP tools take a new `on_mismatch` parameter,
  and `run_iter()` takes the same one. `CheckpointMismatchError` is exported from the package.
- Every AI call logs how long it took, retries included, and the token counts when the
  provider reports them. Phase 2 ends with the total and the average time per node, and the
  run ends with its overall duration.
- `keep_alive_minutes` (default 20) sets how long Ollama keeps the model loaded once idle;
  `-1` keeps it loaded forever, `0` unloads it after each request.

### Changed

- **`ollama_url` is removed from `config.json`.** Ollama is always reached at the
  UnifiedAiClient default address; an old key is reported as unknown and ignored.
- The default `timeout` is 600 seconds, and it now applies to every provider in use,
  including the embedding provider. It was previously registered for Ollama only.
- `temperature` can be `null`, which leaves it to the provider (for Ollama, the Modelfile).
  The GUI has a **Force temperature** toggle.
- The model warm-up runs for every provider and no longer stops the GUI or the CLI when it
  fails: it logs a warning, and the first AI call reports the error.
- UnifiedAiClient is pinned to v0.8.0.
- The per-node log lines no longer repeat the node number, which the progress line above
  them already shows, together with the node's `§` index.
- Ctrl+C in `iter run` ends with a short "Interrupted." message instead of a traceback.

### Fixed

- Setting the context size back to 0 in the GUI now restores the model default. The earlier
  value used to stay in effect until the process restarted.
- The Phase 2 progress counter counts the nodes actually sent, after headings are merged and
  non-text blocks absorbed into sections. It used to show the original block index over the
  merged total, for example `Node 3/2`.
- An unreachable embedding server no longer stalls every node. The first failure is
  reported once and turns cosine similarity off for the rest of the run; previously each
  node waited through the retries twice.

## [0.1.2] - 2026-08-30

### Added

- Save the model settings from the GUI. A **Save as default** button in the sidebar writes the
  current provider, model, embedding, temperature, timeout and context size to `./config.json`
  and reports the path it wrote to. Previously those settings lived only for the browser session.
- `timeout` and `context_size` settings, registered with the provider rather than passed per
  call. `context_size` maps to the Ollama context window; leaving it unset keeps the model
  default.
- The Ollama model is now pre-loaded before the first node, so the model-load latency no longer
  lands on the first AI call of a run.
- `config_file` and `preset_file` parameters on both MCP tools. The MCP server previously had no
  way to reach the presets that the CLI and GUI share.
- `Preset`, `AppConfig` and `ProcessingResult` are re-exported from the package root, so a
  library caller needs one import rather than three.
- A test suite under `tests/`, standard library `unittest` only, runnable with
  `python -m unittest discover -s tests`. Every AI call is stubbed, so it needs no network.
- `Windows Integration/launch_gui.bat`, a GUI launcher that starts from the project folder.
- `ARCHITECTURE.md`, `docs/configuration.md`, `docs/presets.md`, `CONTRIBUTING.md` and this file.

### Changed

- **The Phase 2 node separator now uses a plain hyphen** instead of an em dash:
  `# §3 - Cosine similarity: 0.842`, where the hyphen used to be an em dash. Documents
  produced before this release are unaffected, but any tool that parses the separator needs
  updating. GUI labels changed the same way.
- **`part_of_section` no longer fails silently in paragraph mode.** It absorbs a non-text block
  into the section before it, which cannot work where there are no sections, so the GUI now hides
  the option outside section mode and the processor warns when a preset asks for it anyway.
  Previously the option was selectable everywhere and quietly behaved like "pass through".
- Provider settings are registered by `AppConfig.apply_to_client()`, called by each interface
  when its settings become effective, instead of from inside `run_iter()` on every call. The call
  is idempotent, so changing the provider in the GUI between two runs now takes effect while an
  unchanged run re-registers nothing.
- The MCP tool prompt parameters are typed `str | None = None` instead of defaulting to `""`, and
  `node_mode` and `compute_similarity` likewise. An argument left unset no longer overrides the
  value from a preset file.
- `README.md` is now an overview and usage guide; the internal design moved to `ARCHITECTURE.md`
  and the exhaustive field tables to `docs/`.

### Fixed

- **The MCP server no longer runs against the wrong model without saying so.** It resolved
  `config.json` against the working directory of its client, which is rarely the project folder,
  and silently fell back to the built-in defaults when it found nothing. It now logs the
  configuration path it resolved, warns when there is none, and accepts an explicit one.
- **The logger had drifted from the shared reference implementation** in a way that routed
  compliant calls to stdout: the `mcp` backend had been renamed `stderr` and the `silent` backend
  deleted, so `set_backend("mcp")` and `set_backend("silent")` both fell through to the console
  backend. Under `iter mcp`, stdout carries the JSON-RPC framing. All four backends are restored.
- `AppConfig.load()` and `Preset.load()` warn about the JSON keys they ignore instead of dropping
  them silently, so a stale preset losing a field is now visible.
- The GUI process never called `silence_sdks()`, because `iter gui` spawns a separate process
  that does not go through `main()`. Verbose `google_genai` and `httpx` logging leaked into the
  GUI console for the whole session.
- `iter gui` no longer exits with a raw traceback when the launch fails; it reports the error and
  exits with status 1, like `iter run` already did.
- The CLI writes its fatal errors through the logger, so they carry the same prefix as every
  other message.

### Removed

- Three unused fragments in `gui/app.py`: an unused `json` import, an assignment whose value was
  never read, and a Streamlit placeholder that was never written to.

## [0.1.1] - 2026-06-27

### Fixed

- Resolve paths against the workspace folder, so the CLI and the Windows Explorer integration
  agree on where `Presets/` and `config.json` live.
- Refine the preflight step of the translation presets.
- Update the pinned UnifiedAiClient dependency.

### Changed

- README layout corrections.

## [0.1.0] - 2026-06-27

### Added

- First release: apply an LLM prompt to every paragraph or section node of a Markdown document,
  in three optional phases (preprocessing, node processing, postprocessing).
- Four interfaces over one pipeline: `iter run` (CLI), `iter mcp` (MCP server on stdio),
  `iter gui` (Streamlit), and direct library import.
- Presets saved as JSON in `Presets/`.
- Windows File Explorer context menu integration, generated from the available presets.
- Packaging metadata and the `iter` console script.

[0.1.2]: https://github.com/Kuig/Iter/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Kuig/Iter/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kuig/Iter/releases/tag/v0.1.0

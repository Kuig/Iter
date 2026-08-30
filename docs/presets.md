# Presets

Presets are `.json` files stored in the `Presets/` directory. Each preset saves prompt texts, node mode, and all processing options. Load them in the GUI sidebar or via the `--preset` CLI flag.

## Field reference
Loaded from / saved to `Presets/<name>.json`. Holds the **task configuration**:

| Field | Type | Meaning |
|---|---|---|
| `name` | str | Preset display name |
| `system_prompt` | str | System instruction (overrides AppConfig default when non-empty) |
| `node_mode` | `"paragraph"` \| `"section"` | Granularity of document splitting |
| `preprocessing_prompt` | str | Phase 1 prompt (empty = skip) |
| `node_prompt` | str | Phase 2 prompt (empty = skip) |
| `postprocessing_prompt` | str | Phase 3 prompt (empty = skip) |
| `include_preprocessed_context` | bool | Inject Phase 1 result as prior assistant message in Phase 2 calls |
| `heading_handling` | see below | How to treat `#` heading blocks in Phase 2 |
| `non_text_handling` | see below | How to treat tables / code / images in Phase 2 |
| `compute_similarity` | bool | Compute cosine similarity per node |
| `show_similarity_separator` | bool | Insert `# §N - heading - Cosine similarity: score` between results |
| `postprocess_attach_input` | bool | Append original document to Phase 3 prompt |
| `postprocess_attach_preprocessed` | bool | Append Phase 1 result to Phase 3 prompt |
| `postprocess_attach_results` | bool | Append Phase 2 results to Phase 3 prompt |
| `output_include_preprocessed` | bool | Include Phase 1 result in the output file |
| `output_include_processed` | bool | Include Phase 2 node results in the output file *(default: true)* |
| `output_include_postprocessed` | bool | Include Phase 3 result in the output file |
| *(all three above)* | | Selected sections are appended in order, separated by `---` |
| `save_preprocessed_separately` | bool | Also write Phase 1 result to `<output>_preprocessed.md` |
| `save_postprocessed_separately` | bool | Also write Phase 3 result to `<output>_postprocessed.md` |
| `interlace_input_output` | bool | Save processed nodes interlaced with input nodes *(default: false)* |

**`heading_handling` values:**

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass heading through unchanged *(default)* |
| `"discard"` | Drop the heading entirely |
| `"standalone_nodes"` | Send heading text to the AI as its own node |
| `"merge_next"` | Prepend heading to the next text block before sending to AI |

**`non_text_handling` values** (tables, code blocks, images):

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass block through unchanged *(default)* |
| `"discard"` | Drop the block entirely |
| `"standalone_nodes"` | Send block to the AI as its own node |
| `"part_of_section"` | Absorb block into the surrounding text section *(section mode only)* |

## Adding a field

A `Preset` field is mirrored by hand in several places, and missing one makes the GUI drop
the value silently. The checklist lives in `CLAUDE.md`, next to the code that has to change.

## Forward compatibility

`Preset.load()` ignores keys it does not recognise, so a preset written by an older version
still loads. It logs a warning naming each ignored key, so the loss is never silent.

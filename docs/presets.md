# Presets

Presets are `.json` files stored in the `Presets/` directory. Each preset saves prompt texts, node mode, and all processing options. Load them in the GUI sidebar or via the `--preset` CLI flag.

## Field reference
Loaded from / saved to `Presets/<name>.json`. Holds the **task configuration**:

| Field | Type | Meaning |
|---|---|---|
| `name` | str | Preset display name |
| `system_prompt` | str | System instruction (overrides AppConfig default when non-empty) |
| `node_mode` | `"paragraph"` \| `"section"` \| `"file"` | Granularity of document splitting; `"file"` sends each input file whole, as one node |
| `preprocessing_prompt` | str | Phase 1 prompt (empty = skip) |
| `node_prompt` | str | Phase 2 prompt (empty = skip) |
| `postprocessing_prompt` | str | Phase 3 prompt (empty = skip) |
| `postprocessing_scope` | `"file"` \| `"corpus"` | For a folder input: one Phase 3 call per file *(default)*, or one for the whole folder. See below |
| `include_preprocessed_context` | bool | Inject Phase 1 result as prior assistant message in Phase 2 calls |
| `heading_handling` | see below | How to treat `#` heading blocks in Phase 2 |
| `non_text_handling` | see below | How to treat tables / code blocks in Phase 2 |
| `image_handling` | see below | How to treat standalone image blocks in Phase 2 |
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
| `output_layout` | `"mirror"` \| `"merged"` | For a folder input: one output file per input file in an output folder *(default)*, or a single output file. See below |

**`heading_handling` values:**

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass heading through unchanged *(default)* |
| `"discard"` | Drop the heading entirely |
| `"standalone_nodes"` | Send heading text to the AI as its own node |
| `"merge_next"` | Prepend heading to the next text block before sending to AI |

**`non_text_handling` values** (tables, code blocks):

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass block through unchanged *(default)* |
| `"discard"` | Drop the block entirely |
| `"standalone_nodes"` | Send block to the AI as its own node |
| `"part_of_section"` | Absorb block into the surrounding text section *(section mode only)* |

**`image_handling` values** (a block holding only `![alt](src)`; a block holding only a link is text):

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass the image Markdown through unchanged *(default)* |
| `"discard"` | Drop the image entirely |
| `"standalone_nodes"` | Send the image to the AI as its own node: the prompt carries the Markdown, and the image file is attached |
| `"part_of_section"` | Absorb the image Markdown into the surrounding text section, without attaching the file *(section mode only)* |

With `"standalone_nodes"`, `src` is resolved before any node is sent:

- A relative path is resolved against the input file's folder, or the working directory when the input is raw text.
- An `http(s)` URL is downloaded to a temporary folder, deleted when Phase 2 ends.
- A missing file or a failed download stops the run with an error at the start of Phase 2.

The model has to accept images; the attachment is passed to it as is. With `compute_similarity`, the
original side of the comparison embeds the image together with its Markdown, so it needs a multimodal
embedding model. The image's bytes are part of the node's progress-file key: a changed image is sent
again on resume.

In `"file"` mode there are no structural nodes, so the three handling fields have no effect and
no image is attached.

A thematic break (`---`, `***`, `___` alone on a line) is a node of its own, which also ends a
section: it is always passed through unchanged, whatever the three handling fields say. A heading
underlined with `===` or `---` (setext) is a heading like `# Title` and follows `heading_handling`.
In every mode, a node with no letter or digit (and no attached image) is never sent to the AI; it
is passed through unchanged.

## Folder input

When the input is a folder, its Markdown files (`.md`, `.markdown`) directly inside it are processed
in natural order of their names (`2.md` before `10.md`); subfolders are not read. Every file is
parsed on its own, so no node, section, `merge_next` or `part_of_section` ever spans two files.
Phase 1 always runs once per file, and each node receives the Phase 1 result of its own file. With
a single file as input, `postprocessing_scope` and `output_layout` have no effect.

**`postprocessing_scope` values:**

| Value | Behaviour |
|---|---|
| `"file"` | One Phase 3 call per file; the attachments are that file's input, Phase 1 result and node results *(default)* |
| `"corpus"` | One Phase 3 call for the whole folder; each attachment joins every file's part under a `# File: <name>` header |

With `"corpus"`, `postprocess_attach_input` attaches every input file to one call, which rarely
fits the model's context: the run logs a warning.

**`output_layout` values:**

| Value | The output path is | Content |
|---|---|---|
| `"mirror"` | a folder | One file per input file, with the same name, each composed by `output_include_*` from that file's own results *(default)* |
| `"merged"` | a file | Every input file in order, each under a `# File: <name>` header and composed by `output_include_*`; a file with nothing selected is left out |

In the merged layout, a per-file Phase 3 result follows the node results of its file, and a
corpus-wide Phase 3 result comes last, after a `---` rule.

Where the other results go:

| Case | Written to |
|---|---|
| Mirror, `postprocessing_scope="corpus"` | `<output folder>_postprocessed.md`, beside the folder, whenever Phase 3 ran |
| Mirror, `save_*_separately` | `<name>_preprocessed.md` / `<name>_postprocessed.md` per file, inside the output folder |
| Merged, `save_*_separately` | `<output>_preprocessed.md` / `<output>_postprocessed.md`, with every file under its header |

The output may be neither the input folder nor directly inside it: mirrored files would overwrite
the originals, and any Markdown written there would be read as part of the corpus by the next run.
The run stops with an error before any AI call.

## Adding a field

A `Preset` field is mirrored by hand in several places, and missing one makes the GUI drop
the value silently. The checklist lives in `CLAUDE.md`, next to the code that has to change.

## Forward compatibility

`Preset.load()` ignores keys it does not recognise, so a preset written by an older version
still loads. It logs a warning naming each ignored key, so the loss is never silent.

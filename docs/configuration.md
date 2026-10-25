# Configuration

Iter reads two JSON files from the **current working directory**, falling back to the
package root for `config.json`. Both are gitignored; the committed `.example` files are
the templates. See [ARCHITECTURE.md](../ARCHITECTURE.md) for why paths resolve this way.

### `config.json`

```json
{
  "provider": "ollama",
  "model_name": "gemma4:12b",
  "embedding_provider": "ollama",
  "embedding_model": "bge-m3",
  "temperature": null,
  "thinking": false,
  "timeout": 600,
  "context_size": null,
  "keep_alive_minutes": 20
}
```

### `secrets.json` (do not commit)

```json
{
  "google_api_key": "your-key-here",
  "anthropic_api_key": "your-key-here"
}
```

---

## Field reference (`AppConfig`)
Loaded from `config.json`. Holds the **model settings** that are not saved in presets:

| Field | Default | Meaning |
|---|---|---|
| `provider` | `"ollama"` | AI provider name |
| `model_name` | `"gemma4:12b"` | Main generation model |
| `embedding_provider` | `"ollama"` | AI provider name for embeddings |
| `embedding_model` | `"bge-m3"` | Model used for cosine similarity |
| `temperature` | `null` | Sampling temperature. `null` leaves it to the provider (for Ollama, the Modelfile) |
| `thinking` | `false` | Enable extended thinking / reasoning mode where the provider supports it |
| `timeout` | `600` | Timeout in seconds for each attempt of an AI call. Failed attempts are retried by UnifiedAiClient, so one call can take several times this long |
| `context_size` | `null` | Ollama context window (`num_ctx`). `null` leaves the model default |
| `keep_alive_minutes` | `20` | How long Ollama keeps the model loaded once idle. `-1` keeps it forever, `0` unloads it after each request |

Ollama is always reached at the UnifiedAiClient default address, `http://localhost:11434`.

## Saving from the GUI

The sidebar exposes every one of these fields. **Save as default** writes them to
`./config.json` in the workspace folder, and reports the path it wrote. Loading falls back
to the package root, but saving never writes inside an installed package.

## When settings take effect

Provider settings are registered with UnifiedAiClient by `AppConfig.apply_to_client()`,
which each interface calls when its settings become effective: at startup for the CLI and
the MCP tools, and on each Run for the GUI, where the user can change them between runs.
The timeout is registered with both the generation and the embedding provider, and the
generation model is then warmed up. A failed warm-up is reported as a warning; the first AI
call reports the actual error. The call is idempotent, so an unchanged run re-registers
nothing and skips the warm-up.

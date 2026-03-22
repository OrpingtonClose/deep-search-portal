# TODO: Model Management Separation

## Why this exists

Adding/removing models from Open WebUI required manual DB manipulation, admin UI clicks,
and knowledge of non-obvious internal quirks. This caused repeated outages where models
disappeared from the dropdown. This document captures the production state and lists
concrete tasks to make model management a single-script operation.

## Root cause of the visibility bug (March 2026)

Open WebUI's `utils/models.py:get_filtered_models()` silently drops any model whose
`info` field is missing from the merged model dict. The `info` field is only attached
when `base_model_id IS NULL` in the `model` table. Our proxy models had
`base_model_id = '<own id>'` (self-referencing) instead of `NULL`, so Open WebUI never
merged the DB model info onto the provider model, and the access-control filter dropped
them. Additionally, non-admin users need an `access_grant` row with `principal_id = '*'`
to see models.

**Four places must agree for a model to appear in the dropdown:**

1. Provider URL in `config.data -> openai.OPENAI_API_BASE_URLS` (DB overrides env vars)
2. API key in `config.data -> openai.OPENAI_API_KEYS`
3. `model` table row: `is_active = 1`, `base_model_id = NULL`
4. `access_grant` table row: `resource_id = '<model-id>'`, `principal_id = '*'`, `permission = 'read'`

If any of these is wrong, the model silently vanishes. No error is logged.

## Current production state (as of 2026-03-22)

### Running services (Vast.ai VM: ssh5.vast.ai:18770)

| Service | Screen session | Port | Script |
|---|---|---|---|
| Open WebUI | `owui` | 3000 | `/opt/start_openwebui.sh` |
| SearXNG | `searxng` | 8888 | python -m searx.webapp |
| Thinking Proxy | `thinking-proxy` | 9100 | `/opt/thinking_proxy.py` |
| Deep Research Proxy | `deep-research` | 9200 | `/opt/deep_research_proxy.py` |
| Persistent Proxy | `persistent-proxy` | 9300 | `/opt/persistent_deep_research_proxy.py` |
| Cloudflare Tunnel | `cftunnel` | - | cloudflared tunnel run |
| Knowledge Engine | `knowledge-engine` | - | - |
| Veritas | `veritas` | - | - |

### Active models in dropdown

| Model ID | Display Name | Provider URL | Port |
|---|---|---|---|
| `mistral-large-thinking` | Mistral Large (Thinking) | http://localhost:9100/v1 | 9100 |
| `persistent-miroflow` | Persistent MiroFlow | http://localhost:9300/v1 | 9300 |

### DB provider layout (6 providers, indices 0-5)

| Index | URL | Enabled | model_ids filter |
|---|---|---|---|
| 0 | openrouter.ai | No | [] |
| 1 | venice.ai | No | [] |
| 2 | together.xyz | No | [] |
| 3 | perplexity.ai | No | [] |
| 4 | localhost:9100 | Yes | ["mistral-large-thinking"] |
| 5 | localhost:9300 | Yes | ["persistent-miroflow"] |

### Inactive model table entries (22 legacy models with `is_active = 0`)

These are left over from previous provider configs (OpenRouter, Venice, RunPod, etc).
They are harmless but clutter the DB.

## Tasks

### Task 1: Create `scripts/models.yaml`

A YAML file that is the single source of truth for which models are exposed.

```yaml
models:
  mistral-large-thinking:
    display_name: "Mistral Large (Thinking)"
    provider_url: "http://localhost:9100/v1"
    api_key: "not-needed"
    public: true

  persistent-miroflow:
    display_name: "Persistent MiroFlow"
    provider_url: "http://localhost:9300/v1"
    api_key: "not-needed"
    public: true

disabled_providers:
  - url: "https://openrouter.ai/api/v1"
    key_env: "OPENROUTER_API_KEY"
  - url: "https://api.venice.ai/api/v1"
    key_env: "VENICE_API_KEY"
  - url: "https://api.together.xyz/v1"
    key_env: "TOGETHER_API_KEY"
  - url: "https://api.perplexity.ai"
    key_env: "PERPLEXITY_API_KEY"
```

### Task 2: Create `scripts/sync_models.py`

Python script that reads `models.yaml` and writes to the Open WebUI SQLite DB.
Must handle all four places listed above:

- `config` table: provider URLs, keys, configs (with correct indices)
- `model` table: upsert rows with `base_model_id = NULL`, `is_active = 1`
- `access_grant` table: upsert wildcard read grants for public models
- Deactivate models not in YAML

Must be **idempotent** (safe to run multiple times).
Must print a summary of what changed.
Must NOT require Open WebUI restart (DB changes take effect within 1s due to cache TTL).

DB path: `/opt/openwebui-data/webui.db`

### Task 3: Integrate into `scripts/startup.sh`

After `wait_for_health "http://localhost:3000" "Open WebUI" 60`, add:
```bash
python3 /opt/sync_models.py /opt/models.yaml
```

### Task 4: Simplify `scripts/start_openwebui.sh`

Remove the hand-written `OPENAI_API_BASE_URLS`, `OPENAI_API_KEYS`,
`OPENAI_API_CONFIGS` exports. These are now managed by `sync_models.py`.
Keep only: OAuth config, SearXNG, DATA_DIR, WEBUI_URL, access control.

Set `ENABLE_OPENAI_API=true` and minimal placeholder URLs for first boot
(sync_models.py will overwrite on first run).

### Task 5: Clean up legacy model rows

Delete or deactivate the 22 legacy `model` table rows (cu-venice-uncensored,
venice-roleplay, hermes-3-70b, etc.) and their associated `access_grant` rows.
Can be done in `sync_models.py` as part of the "deactivate models not in YAML" logic.

### Task 6: Deploy and verify

1. Copy `sync_models.py` and `models.yaml` to `/opt/` on the VM
2. Run `python3 /opt/sync_models.py /opt/models.yaml`
3. Verify both models appear in dropdown for admin and non-admin users
4. Test adding a dummy model to YAML, running sync, verify it appears
5. Remove dummy model, run sync, verify it disappears

## How to add a new model (after this is implemented)

```bash
# 1. Edit scripts/models.yaml — add your model entry
# 2. Start the proxy if it's a new one
# 3. Run:
python3 /opt/sync_models.py /opt/models.yaml
# Done. Model appears in dropdown for all users.
```

## Files on the VM

| Path | What it is |
|---|---|
| `/opt/start_openwebui.sh` | Open WebUI startup (env vars + exec) |
| `/opt/startup.sh` | Master startup (all services) |
| `/opt/.env` | Secrets (API keys, OAuth credentials) |
| `/opt/openwebui-data/webui.db` | Open WebUI SQLite database |
| `/opt/thinking_proxy.py` | Thinking proxy (port 9100) |
| `/opt/persistent_deep_research_proxy.py` | Persistent proxy (port 9300) |
| `/opt/deep_research_proxy.py` | Deep research proxy (port 9200) |

## Key Open WebUI internals (for reference)

- Version: 0.8.8 (0.8.10 available)
- DB config key: `config` table, `id = '1'`, `data` column (JSON)
- Model cache TTL: 1 second (`MODELS_CACHE_TTL`)
- Env vars only used on first boot; DB `config` table overrides after that
- `get_filtered_models()` in `utils/models.py` is the final filter before API response
- `base_model_id = NULL` means "this model IS the base model" (info gets attached)
- `base_model_id = '<some-id>'` means "this is a custom wrapper" (different code path)

#!/usr/bin/env python3
"""sync_models.py — Idempotent model management for Open WebUI.

Reads a models.yaml file and writes to the Open WebUI SQLite database,
ensuring all four places agree:

  1. Provider URLs in config.data -> openai.OPENAI_API_BASE_URLS
  2. API keys in config.data -> openai.OPENAI_API_KEYS
  3. model table rows: is_active=1, base_model_id=NULL
  4. access_grant table rows: principal_id='*', permission='read'

Models NOT in the YAML are deactivated (is_active=0) and their grants removed.

Usage:
    python3 scripts/sync_models.py scripts/models.yaml
    python3 scripts/sync_models.py scripts/models.yaml --db-path /opt/openwebui-data/webui.db
    python3 scripts/sync_models.py scripts/models.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import yaml

DEFAULT_DB_PATH = "/opt/openwebui-data/webui.db"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_db(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# 1. Provider config (config table)
# ---------------------------------------------------------------------------

def sync_providers(
    conn: sqlite3.Connection,
    models: dict,
    disabled_providers: list[dict],
    dry_run: bool,
) -> list[str]:
    """Build provider URL/key/config lists and write to the config table.

    Returns list of change descriptions.
    """
    changes: list[str] = []

    # Build ordered provider list: disabled first, then active model providers
    urls: list[str] = []
    keys: list[str] = []
    configs: dict[str, dict] = {}

    # Disabled providers (indices 0..N-1)
    for dp in disabled_providers:
        urls.append(dp["url"])
        key_env = dp.get("key_env", "")
        keys.append(os.environ.get(key_env, "not-set") if key_env else "not-set")

    # Active model providers (indices N..N+M-1)
    active_offset = len(disabled_providers)
    model_ids_by_url: dict[str, list[str]] = {}
    model_keys_by_url: dict[str, str] = {}

    for model_id, model_cfg in models.items():
        provider_url = model_cfg["provider_url"]
        if provider_url not in model_ids_by_url:
            model_ids_by_url[provider_url] = []
            # Resolve API key: api_key_env (from env var) takes precedence over api_key (literal)
            key_env = model_cfg.get("api_key_env", "")
            if key_env:
                model_keys_by_url[provider_url] = os.environ.get(key_env, "not-set")
            else:
                model_keys_by_url[provider_url] = model_cfg.get("api_key", "not-needed")
        model_ids_by_url[provider_url].append(model_id)

    for i, (provider_url, model_ids) in enumerate(model_ids_by_url.items()):
        idx = active_offset + i
        urls.append(provider_url)
        keys.append(model_keys_by_url[provider_url])
        configs[str(idx)] = {"enable": True, "model_ids": model_ids}

    urls_str = ";".join(urls)
    keys_str = ";".join(keys)
    configs_json = json.dumps(configs)

    # Read current config
    cur = conn.execute("SELECT data FROM config WHERE id = '1'")
    row = cur.fetchone()
    if row is None:
        changes.append("ERROR: No config row found (id=1)")
        return changes

    config_data = json.loads(row["data"])
    openai = config_data.get("openai", {})
    old_urls = openai.get("OPENAI_API_BASE_URLS", "")
    old_keys = openai.get("OPENAI_API_KEYS", "")
    old_configs = json.dumps(openai.get("OPENAI_API_CONFIGS", {}))

    if old_urls != urls_str:
        changes.append(f"  providers: URLs updated ({len(urls)} providers)")
    if old_keys != keys_str:
        changes.append(f"  providers: API keys updated")
    if old_configs != configs_json:
        changes.append(f"  providers: configs updated -> {configs_json}")

    if changes and not dry_run:
        openai["OPENAI_API_BASE_URLS"] = urls_str
        openai["OPENAI_API_KEYS"] = keys_str
        openai["OPENAI_API_CONFIGS"] = json.loads(configs_json)
        openai["ENABLE_OPENAI_API"] = True
        config_data["openai"] = openai
        conn.execute(
            "UPDATE config SET data = ? WHERE id = '1'",
            (json.dumps(config_data),),
        )

    if not changes:
        changes.append("  providers: no changes needed")

    return changes


# ---------------------------------------------------------------------------
# 2. Model table
# ---------------------------------------------------------------------------

def sync_model_rows(
    conn: sqlite3.Connection,
    models: dict,
    dry_run: bool,
) -> list[str]:
    """Upsert model rows and deactivate models not in YAML.

    Returns list of change descriptions.
    """
    changes: list[str] = []
    now = int(time.time())
    yaml_ids = set(models.keys())

    for model_id, model_cfg in models.items():
        display_name = model_cfg.get("display_name", model_id)
        description = model_cfg.get("description", "")

        cur = conn.execute("SELECT id, is_active, name FROM model WHERE id = ?", (model_id,))
        existing = cur.fetchone()

        meta = json.dumps({
            "description": description,
            "profile_image_url": "",
            "capabilities": {"vision": False},
        })
        # Build params: include system prompt and sampling params if configured
        params_dict: dict = {}
        if model_cfg.get("system_prompt"):
            params_dict["system"] = model_cfg["system_prompt"]
        if model_cfg.get("temperature") is not None:
            params_dict["temperature"] = model_cfg["temperature"]
        if model_cfg.get("top_p") is not None:
            params_dict["top_p"] = model_cfg["top_p"]
        params = json.dumps(params_dict)

        base_model = model_cfg.get("base_model_id")

        if existing is None:
            changes.append(f"  model: INSERT {model_id} ({display_name})")
            if not dry_run:
                conn.execute(
                    """INSERT INTO model (id, name, meta, params, base_model_id, is_active, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                    (model_id, display_name, meta, params, base_model, now, now),
                )
        else:
            if not existing["is_active"]:
                changes.append(f"  model: ACTIVATE {model_id}")
                if not dry_run:
                    conn.execute(
                        "UPDATE model SET is_active = 1, base_model_id = ?, name = ?, meta = ?, params = ?, updated_at = ? WHERE id = ?",
                        (base_model, display_name, meta, params, now, model_id),
                    )
            else:
                # Ensure base_model_id, name, and params are correct
                cur2 = conn.execute(
                    "SELECT base_model_id, name, params FROM model WHERE id = ?", (model_id,)
                )
                row = cur2.fetchone()
                needs_update = False
                if row["base_model_id"] != base_model:
                    changes.append(f"  model: FIX base_model_id={base_model} for {model_id}")
                    needs_update = True
                if row["name"] != display_name:
                    changes.append(f"  model: UPDATE name {model_id} -> {display_name}")
                    needs_update = True
                if row["params"] != params:
                    changes.append(f"  model: UPDATE params for {model_id}")
                    needs_update = True
                if needs_update and not dry_run:
                    conn.execute(
                        "UPDATE model SET base_model_id = ?, name = ?, meta = ?, params = ?, updated_at = ? WHERE id = ?",
                        (base_model, display_name, meta, params, now, model_id),
                    )

    # Deactivate models not in YAML
    cur = conn.execute("SELECT id, name FROM model WHERE is_active = 1")
    for row in cur.fetchall():
        if row["id"] not in yaml_ids:
            changes.append(f"  model: DEACTIVATE {row['id']} ({row['name']})")
            if not dry_run:
                conn.execute(
                    "UPDATE model SET is_active = 0, updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )

    if not any("model:" in c for c in changes):
        changes.append("  model: no changes needed")

    return changes


# ---------------------------------------------------------------------------
# 3. Access grants
# ---------------------------------------------------------------------------

def sync_access_grants(
    conn: sqlite3.Connection,
    models: dict,
    dry_run: bool,
) -> list[str]:
    """Upsert wildcard read grants for public models, remove grants for non-YAML models.

    Returns list of change descriptions.
    """
    changes: list[str] = []

    for model_id, model_cfg in models.items():
        if not model_cfg.get("public", True):
            continue

        resource_id = f"model:{model_id}"
        cur = conn.execute(
            "SELECT id FROM access_grant WHERE resource_id = ? AND principal_id = '*' AND permission = 'read'",
            (resource_id,),
        )
        if cur.fetchone() is None:
            changes.append(f"  grant: ADD wildcard read for {model_id}")
            if not dry_run:
                conn.execute(
                    """INSERT INTO access_grant (resource_id, resource_type, principal_id, principal_type, permission, created_at)
                       VALUES (?, 'model', '*', 'role', 'read', ?)""",
                    (resource_id, int(time.time())),
                )

    # Remove grants for models not in YAML
    yaml_resource_ids = {f"model:{mid}" for mid, cfg in models.items() if cfg.get("public", True)}
    cur = conn.execute(
        "SELECT id, resource_id FROM access_grant WHERE principal_id = '*' AND permission = 'read' AND resource_type = 'model'"
    )
    for row in cur.fetchall():
        if row["resource_id"] not in yaml_resource_ids:
            changes.append(f"  grant: REMOVE wildcard read for {row['resource_id']}")
            if not dry_run:
                conn.execute("DELETE FROM access_grant WHERE id = ?", (row["id"],))

    if not any("grant:" in c for c in changes):
        changes.append("  grant: no changes needed")

    return changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync models.yaml to Open WebUI DB")
    parser.add_argument("yaml_path", help="Path to models.yaml")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to webui.db")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    args = parser.parse_args()

    cfg = load_yaml(args.yaml_path)
    models = cfg.get("models", {})
    disabled_providers = cfg.get("disabled_providers", [])

    if not models:
        print("WARNING: No models defined in YAML", file=sys.stderr)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Syncing {len(models)} models from {args.yaml_path}")
    print(f"DB: {args.db_path}")
    print()

    conn = get_db(args.db_path)

    all_changes: list[str] = []

    print("[1/3] Syncing provider config...")
    changes = sync_providers(conn, models, disabled_providers, args.dry_run)
    all_changes.extend(changes)
    for c in changes:
        print(c)

    print("\n[2/3] Syncing model rows...")
    changes = sync_model_rows(conn, models, args.dry_run)
    all_changes.extend(changes)
    for c in changes:
        print(c)

    print("\n[3/3] Syncing access grants...")
    changes = sync_access_grants(conn, models, args.dry_run)
    all_changes.extend(changes)
    for c in changes:
        print(c)

    if not args.dry_run:
        conn.commit()
        print("\nCommitted to database.")
    else:
        print("\n[DRY RUN] No changes written.")

    conn.close()

    real_changes = [c for c in all_changes if "no changes needed" not in c and "ERROR" not in c]
    print(f"\nSummary: {len(real_changes)} change(s) applied to {len(models)} models.")


if __name__ == "__main__":
    main()

"""
Persistent API key storage in instance/api_keys.json.

Keys survive restarts and are merged individually — adding one key never
overwrites others.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

API_KEYS_FILE = os.path.join(os.path.dirname(__file__), "..", "instance", "api_keys.json")
API_KEYS_FILE = os.path.normpath(API_KEYS_FILE)


def load_api_keys() -> Dict[str, str]:
    """Return all saved API keys from the JSON file (empty dict if none)."""
    if not os.path.exists(API_KEYS_FILE):
        return {}
    try:
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.error("Failed to load API keys: %s", exc)
        return {}


def _write_keys(keys: Dict[str, str]) -> bool:
    try:
        os.makedirs(os.path.dirname(API_KEYS_FILE), exist_ok=True)
        with open(API_KEYS_FILE, "w") as f:
            json.dump(keys, f, indent=2)
        return True
    except IOError as exc:
        logger.error("Failed to write API keys file: %s", exc)
        return False


def save_api_key(platform: str, key_value: str) -> bool:
    """
    Save a single platform key without touching others.
    Empty key_value clears the entry.
    """
    existing = load_api_keys()
    key_value = key_value.strip() if key_value else ""
    if key_value:
        existing[platform] = key_value
        logger.info("API key saved for platform: %s", platform)
    else:
        existing.pop(platform, None)
        logger.info("API key cleared for platform: %s", platform)
    return _write_keys(existing)


def delete_api_key(platform: str) -> bool:
    """Remove a single platform key from persistent storage."""
    existing = load_api_keys()
    if platform not in existing:
        return False
    del existing[platform]
    logger.info("API key deleted for platform: %s", platform)
    return _write_keys(existing)


def apply_keys_to_environment(platform_env_map: Dict[str, str]) -> int:
    """
    Load saved keys and inject them into os.environ.
    Only sets keys not already present (env/systemd values take priority).
    Returns the number of keys applied.
    """
    saved = load_api_keys()
    count = 0
    for platform, env_var in platform_env_map.items():
        if not env_var:
            continue
        key_value = saved.get(platform)
        if key_value and not os.environ.get(env_var):
            os.environ[env_var] = key_value
            count += 1
            logger.debug("Restored API key for %s → %s", platform, env_var)
    if count:
        logger.info("Restored %d API key(s) from persistent storage", count)
    return count


def migrate_env_keys_to_file(platform_env_map: Dict[str, str]) -> int:
    """
    One-time migration: persist any keys already in environment (e.g. from
    systemd Environment= lines) into the JSON file so they survive future
    restarts even if the service file changes.
    Returns count of newly migrated keys.
    """
    existing = load_api_keys()
    migrated = 0
    for platform, env_var in platform_env_map.items():
        if not env_var:
            continue
        env_value = os.environ.get(env_var)
        if env_value and platform not in existing:
            existing[platform] = env_value
            migrated += 1
            logger.info("Migrated env key for platform: %s", platform)
    if migrated:
        _write_keys(existing)
        logger.info("Migrated %d API key(s) from environment to persistent storage", migrated)
    return migrated

"""Tests for kiro_crew.mcp_gateway.secret_uri — cleanup contract.

These tests pin the parent-process cleanup contract for resolved
``secret://`` URIs. The contract exists in the ``resolve_secret_uris``
docstring and is honoured by ``gatewayd._acquire_backend`` today, but it
was not previously locked by a test, so a refactor could silently leak
plaintext into the long-running gateway process's Python heap.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import kiro_crew.sel as sel_mod
from kiro_crew.mcp_gateway.secret_uri import clear_resolved_secrets, resolve_secret_uris
from kiro_crew.sel import SecurityEventLog


@pytest.fixture()
def vault_dir(tmp_path: Path) -> Path:
    """Create a temporary vault with test secrets."""
    from kiro_crew.secrets import SecretVault

    vault = SecretVault(tmp_path)
    vault._set_sync("MY_API_KEY", "sk-live-abc123")
    vault._set_sync("DB_PASS", "super-secret-password")
    return tmp_path


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the SEL singleton between tests.

    Preserves the prior instance/initialized state across each test rather
    than unconditionally nulling it: a real production singleton may carry
    a running writer thread, and discarding the reference while the thread
    is still alive leaks it. Each test that needs a fresh log creates its
    own ``SecurityEventLog(base_dir=..., sync=True)`` so the inline path
    never spawns a writer in the first place.
    """
    prior_instance = SecurityEventLog._instance
    prior_initialized = SecurityEventLog._initialized
    SecurityEventLog._instance = None
    SecurityEventLog._initialized = False
    yield
    SecurityEventLog._instance = prior_instance
    SecurityEventLog._initialized = prior_initialized


class TestClearResolvedSecrets:
    """The ``clear_resolved_secrets`` helper is the single pop helper callers
    must use after spawn. It mirrors what ``gatewayd`` used to inline."""

    def test_removes_plaintext_for_every_resolved_key(self, vault_dir: Path) -> None:
        env = {"API_KEY": "secret://MY_API_KEY", "HOST": "localhost"}
        resolved, secret_keys = resolve_secret_uris(env, vault_dir)
        assert "API_KEY" in resolved  # pre-condition: resolution happened

        clear_resolved_secrets(resolved, secret_keys)

        assert "API_KEY" not in resolved
        assert resolved == {"HOST": "localhost"}

    def test_is_noop_when_secret_keys_empty(self) -> None:
        env = {"HOST": "localhost", "PORT": "5432"}
        clear_resolved_secrets(env, set())
        assert env == {"HOST": "localhost", "PORT": "5432"}

    def test_does_not_mutate_caller_when_key_missing(self) -> None:
        """If ``secret_keys`` references a key the caller already popped, the
        helper must not raise — it must use ``pop(..., default=None)`` so a
        caller that pre-cleaned is still safe."""
        env: dict[str, str] = {"HOST": "localhost"}
        clear_resolved_secrets(env, {"GONE"})
        assert env == {"HOST": "localhost"}

    def test_plaintext_value_does_not_survive_across_helpers(self, vault_dir: Path) -> None:
        """The full lifecycle: resolve then clear. Plaintext must not be
        observable in the returned dict after the cleanup helper runs."""
        env = {"API_KEY": "secret://MY_API_KEY", "PASSWORD": "secret://DB_PASS"}
        resolved, secret_keys = resolve_secret_uris(env, vault_dir)

        clear_resolved_secrets(resolved, secret_keys)

        # No key that was a secret:// reference may carry the vault plaintext
        for key in ("API_KEY", "PASSWORD"):
            assert key not in resolved
        assert "sk-live-abc123" not in json.dumps(resolved)
        assert "super-secret-password" not in json.dumps(resolved)


class TestResolveSecretUrisEmitsSelEvent:
    """``resolve_secret_uris`` MUST emit a SEL event whenever it resolves one
    or more ``secret://`` URIs. The event carries the resolved KEY NAMES
    only — never values — so the SEL becomes a forensic record of "which
    env-var names held a secret in this spawn" without leaking plaintext to
    the audit log."""

    def test_emits_one_event_when_at_least_one_secret_resolved(
        self, vault_dir: Path, tmp_path: Path
    ) -> None:
        # Redirect SEL storage into the per-test temp dir before any call
        # touches it. This mirrors the test_sel.py fixture pattern.
        sel_dir = tmp_path / "sel"
        log = SecurityEventLog(base_dir=sel_dir, sync=True)

        env = {"API_KEY": "secret://MY_API_KEY", "HOST": "localhost"}
        with patch.object(sel_mod, "sel", return_value=log):
            resolve_secret_uris(
                env,
                vault_dir,
                caller_identity="gatewayd:test",
                source="mcp-gateway",
            )

        sel_file = sel_dir / "security_events.jsonl"
        assert sel_file.exists()
        lines = sel_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["event_type"] == "secret_uri_resolved"
        assert record["source"] == "mcp-gateway"
        assert record["caller_identity"] == "gatewayd:test"
        # KEY NAMES only — never the plaintext values
        assert record["metadata"]["resolved_keys"] == ["API_KEY"]
        # Timestamp must be a real ISO 8601 string — an empty value would let
        # sel().prune() treat the event as expired and delete the forensic
        # record at the next retention sweep.
        timestamp = record["timestamp"]
        assert timestamp
        datetime.fromisoformat(timestamp)  # raises if not ISO 8601
        # Defence in depth: plaintext must never reach the log
        assert "sk-live-abc123" not in sel_file.read_text(encoding="utf-8")
        assert "MY_API_KEY" not in sel_file.read_text(encoding="utf-8")

    def test_emits_one_event_with_all_resolved_keys(self, vault_dir: Path, tmp_path: Path) -> None:
        sel_dir = tmp_path / "sel"
        log = SecurityEventLog(base_dir=sel_dir, sync=True)

        env = {
            "API_KEY": "secret://MY_API_KEY",
            "PASSWORD": "secret://DB_PASS",
            "HOST": "localhost",
        }
        with patch.object(sel_mod, "sel", return_value=log):
            resolve_secret_uris(
                env,
                vault_dir,
                caller_identity="gatewayd:test",
                source="mcp-gateway",
            )

        record = json.loads((sel_dir / "security_events.jsonl").read_text(encoding="utf-8").strip())
        # Order is not guaranteed (set iteration); compare as sorted list
        assert sorted(record["metadata"]["resolved_keys"]) == ["API_KEY", "PASSWORD"]

    def test_no_event_when_no_secret_uri_resolved(self, vault_dir: Path, tmp_path: Path) -> None:
        sel_dir = tmp_path / "sel"
        log = SecurityEventLog(base_dir=sel_dir, sync=True)

        env = {"HOST": "localhost", "PORT": "5432"}
        with patch.object(sel_mod, "sel", return_value=log):
            resolve_secret_uris(
                env,
                vault_dir,
                caller_identity="gatewayd:test",
                source="mcp-gateway",
            )

        sel_file = sel_dir / "security_events.jsonl"
        assert not sel_file.exists()

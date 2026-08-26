"""Resolve ``secret://NAME`` URIs in MCP server environment variables.

At spawn time, env values matching the ``secret://`` scheme are resolved
against the local :class:`~kiro_crew.secrets.SecretVault`. Resolution is
in-memory only — the sidecar file on disk retains the raw URI template so
the secret is never persisted in plaintext outside the vault.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kiro_crew.secrets import SecretVault
from kiro_crew.sel import SecurityEvent
from kiro_crew.sel import sel as _sel

logger = logging.getLogger(__name__)

_SECRET_URI_RE = re.compile(r"^secret://(.+)$")


def clear_resolved_secrets(env: dict[str, str], secret_keys: set[str]) -> None:
    """Remove resolved secret keys from *env* in place.

    This is the single, named helper that callers MUST use after a
    ``resolve_secret_uris`` call returns, to honour the parent-process
    memory-hygiene contract documented on :func:`resolve_secret_uris`.
    Using a single helper means a regression test can pin the contract and
    a future refactor of the spawn path cannot silently leak plaintext into
    the long-running gateway's Python heap.

    ``pop(..., default=None)`` makes the helper a safe no-op when the
    caller has already removed a key — idempotent, so a defence-in-depth
    double-clean is harmless.
    """
    for key in secret_keys:
        env.pop(key, None)


def resolve_secret_uris(
    env: dict[str, str],
    config_dir: Path,
    *,
    caller_identity: str = "",
    source: str = "",
) -> tuple[dict[str, str], set[str]]:
    """Return a copy of *env* with ``secret://NAME`` values resolved.

    Returns ``(resolved_env, secret_keys)`` where *secret_keys* is the set
    of env-var names that held a ``secret://`` URI and now contain plaintext.
    The caller MUST clear these keys from the returned dict after the child
    process has been spawned (``exec`` copies them into the child's address
    space) so that plaintext secrets do not linger in parent-process memory.
    Use :func:`clear_resolved_secrets` for that step — it is the single
    named helper and the only thing regression tests assert on.

    When one or more ``secret://`` URIs are resolved, this function emits a
    SEL event (``event_type="secret_uri_resolved"``) carrying the RESOLVED
    KEY NAMES — never values — so the audit log becomes a forensic record
    of "which env-var names held a secret in this spawn" without ever
    leaking plaintext. Pass ``caller_identity`` and ``source`` to attribute
    the event; both default to empty strings to keep the signature
    backwards-compatible with existing call sites.

    Non-matching values pass through unchanged. Raises :exc:`ValueError`
    when a referenced secret does not exist in the vault — failing closed
    prevents an MCP server from starting with a missing credential.

    This function is intentionally synchronous: vault reads are local
    filesystem I/O and the caller (gatewayd spawn path) is already in an
    async context that would need ``await asyncio.to_thread(...)`` for a
    blocking call — keeping this sync lets the caller wrap it once.
    """
    vault = SecretVault(config_dir)
    resolved: dict[str, str] = {}
    secret_keys: set[str] = set()

    for key, value in env.items():
        m = _SECRET_URI_RE.match(value)
        if m is None:
            resolved[key] = value
            continue

        secret_name = m.group(1)
        secret_value = vault.get(secret_name)
        if secret_value is None:
            raise ValueError(
                f"MCP server env var {key!r} references secret://{secret_name} "
                f"but no secret named {secret_name!r} exists in the vault. "
                f"Run `kirocrew secrets set {secret_name}` to store it."
            )
        resolved[key] = secret_value.reveal()
        secret_keys.add(key)

    if secret_keys:
        _emit_secret_resolved_event(
            resolved_keys=sorted(secret_keys),
            caller_identity=caller_identity,
            source=source,
        )

    return resolved, secret_keys


def _emit_secret_resolved_event(
    *,
    resolved_keys: list[str],
    caller_identity: str,
    source: str,
) -> None:
    """Best-effort SEL audit on secret resolution. Failures here MUST NOT
    block MCP spawn: a SEL outage is not a credential outage. The audit
    payload carries KEY NAMES only — never values — so the log becomes a
    forensic record without becoming a leak."""
    try:
        event = SecurityEvent(
            event_id=f"sec-uri-{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            event_type="secret_uri_resolved",
            caller_identity=caller_identity,
            agent="kirocrew",
            source=source or "mcp-gateway",
            operation="resolve_secret_uris",
            outcome="completed",
            metadata={"resolved_keys": resolved_keys},
        )
        _sel().log(event)
    except Exception as exc:
        # Defence in depth — the SEL writer should never break the spawn path.
        logger.warning(
            "SEL audit emission for secret_uri_resolved failed; "
            "spawn will proceed without an audit record: %s",
            exc,
        )

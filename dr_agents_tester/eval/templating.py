"""``{env:VAR}`` token substitution for behavioral scenarios (fixture-id injection).

Some scenarios run against pre-provisioned, long-lived DataRobot resources
(e.g. a fixture deployment the predictions scenario scores against) whose ids
differ per account. Scenarios declare the host environment variables they need
in ``requires_env`` and reference them as ``{env:VAR}`` tokens in the prompt,
``env`` values, and string check params.

Enforcement is layered so failures are loud and early:

1. Parse time (offline): every referenced token must be declared in
   ``requires_env``, and declared names must match the ``BEHAVIORAL_``/``DRAT_``
   prefix allowlist — a scenario can never template arbitrary host state such
   as ``DATAROBOT_API_TOKEN`` into a prompt.
2. Run start: missing/empty declared variables abort the whole invocation
   before any agent tokens are spent (``AgentBackend.load_scenarios``).
3. Substitution: an unresolved token raises rather than passing ``{env:...}``
   text through to an agent or a check.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

ENV_TOKEN_RE = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")

#: Allowlist for requires_env entries (see module docstring, layer 1).
ALLOWED_ENV_NAME_RE = re.compile(r"^(BEHAVIORAL_|DRAT_)[A-Z0-9_]+$")


def find_env_tokens(text: str) -> set[str]:
    """Variable names referenced as ``{env:VAR}`` tokens in a string."""
    return set(ENV_TOKEN_RE.findall(text))


def substitute_env_tokens(text: str, host_env: Mapping[str, str]) -> str:
    """Replace every ``{env:VAR}`` token with its host-environment value.

    Raises ValueError when a referenced variable is missing or empty — a
    partially-templated prompt or check param must never survive silently.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        value = host_env.get(name, "")
        if not value:
            raise ValueError(
                f"Environment variable {name!r} (referenced as {{env:{name}}}) is not set"
            )
        return value

    return ENV_TOKEN_RE.sub(_sub, text)

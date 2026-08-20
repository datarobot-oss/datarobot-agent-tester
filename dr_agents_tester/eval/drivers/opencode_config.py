"""Generate the per-run opencode.json wiring OpenCode to the DataRobot LLM Gateway.

The config lands in the run workspace (highest-precedence config file), and
the API key stays an ``{env:...}`` reference so the token never touches disk.
Verified live: the gateway's OpenAI-compatible surface is
``{DATAROBOT_ENDPOINT}/genai/llmgw`` and gateway model ids are
provider-prefixed (e.g. ``anthropic/claude-sonnet-4-6``).
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODEL = "datarobot/anthropic/claude-sonnet-4-6"
PROVIDER_ID = "datarobot"

CONFIG_FILENAME = "opencode.json"


def gateway_base_url(endpoint: str) -> str:
    """Derive the LLM Gateway base URL from a DataRobot API endpoint."""
    return endpoint.rstrip("/") + "/genai/llmgw"


def split_model(model: str) -> tuple[str, str]:
    """Split ``datarobot/anthropic/claude-...`` into (provider, gateway model id)."""
    provider, _, gateway_model = model.partition("/")
    if not gateway_model:
        raise ValueError(
            f"Model must be '<provider>/<gateway-model-id>', got {model!r} (e.g. {DEFAULT_MODEL!r})"
        )
    return provider, gateway_model


def build_opencode_config(
    endpoint: str,
    model: str = DEFAULT_MODEL,
    context_limit: int = 200_000,
    output_limit: int = 64_000,
) -> dict[str, object]:
    """Build the opencode.json dict for a run against the DataRobot LLM Gateway."""
    provider, gateway_model = split_model(model)
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DataRobot LLM Gateway",
                "options": {
                    "baseURL": gateway_base_url(endpoint),
                    "apiKey": "{env:DATAROBOT_API_TOKEN}",
                },
                "models": {
                    gateway_model: {
                        "name": f"{gateway_model} (DataRobot LLM Gateway)",
                        "limit": {"context": context_limit, "output": output_limit},
                    }
                },
            }
        },
    }


def write_run_config(workspace: Path, endpoint: str, model: str = DEFAULT_MODEL) -> Path:
    """Write the run's opencode.json into the workspace; returns its path."""
    config_path = workspace / CONFIG_FILENAME
    config_path.write_text(json.dumps(build_opencode_config(endpoint, model), indent=2) + "\n")
    return config_path

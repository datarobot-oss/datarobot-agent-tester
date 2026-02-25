"""Thin wrapper around litellm for the DataRobot LLM gateway."""

from .config import Config


def call_llm(prompt: str, model: str, config: Config) -> str:
    """Send a prompt to the LLM gateway and return the response text.

    Uses a deployed model (LLM_DEPLOYMENT_ID) when configured, otherwise
    routes directly to the named model via the DataRobot gateway.

    Args:
        prompt: The user prompt to send.
        model: LiteLLM model identifier to use.
        config: Gateway connection settings.

    Returns:
        The model's response as a stripped string.

    Raises:
        RuntimeError: If the model returns an empty response.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("litellm is not installed. Run:\n  uv sync\nor:\n  pip install litellm")

    base_url = config.endpoint.rstrip("/")
    if base_url.endswith("/api/v2"):
        base_url = base_url[: -len("/api/v2")]

    if config.deployment_id:
        api_base = f"{base_url}/api/v2/deployments/{config.deployment_id}/chat/completions"
        call_model = "datarobot/datarobot-deployed-llm"
    else:
        api_base = f"{base_url}/"
        call_model = model

    response = litellm.completion(
        model=call_model,
        messages=[{"role": "user", "content": prompt}],
        api_base=api_base,
        api_key=config.api_key,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(
            f"Model {call_model!r} returned an empty response. "
            "Try a different model via --model / --test-model or the AGENTS_MD_MODEL env var."
        )
    return str(content).strip()

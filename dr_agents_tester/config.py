"""Configuration for the DataRobot LLM gateway connection and model selection."""

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "datarobot/anthropic/claude-opus-4-5-20251101"
DEFAULT_TEST_MODEL = "datarobot/anthropic/claude-sonnet-4-5-20250929"


@dataclass
class Config:
    """LLM gateway connection settings.

    All fields fall back to environment variables so callers can use either:

        config = Config(api_key="mytoken")          # explicit
        config = Config()                            # reads from env / .env
    """

    api_key: str = field(default_factory=lambda: os.environ.get("DATAROBOT_API_TOKEN", ""))
    endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2"
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("AGENTS_MD_MODEL", DEFAULT_MODEL)
    )
    test_model: str = field(
        default_factory=lambda: os.environ.get("AGENTS_MD_TEST_MODEL", DEFAULT_TEST_MODEL)
    )
    deployment_id: str | None = field(
        default_factory=lambda: os.environ.get("LLM_DEPLOYMENT_ID")
    )

    def validate(self) -> None:
        """Raise ValueError if required fields are missing."""
        if not self.api_key:
            raise ValueError(
                "DATAROBOT_API_TOKEN is not set. "
                "Add it to .env or export it as an environment variable."
            )

"""Shared pytest fixtures."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_agents_md(fixtures_dir: Path) -> str:
    return (fixtures_dir / "sample_agents_md.md").read_text()


@pytest.fixture
def sample_skill(fixtures_dir: Path) -> str:
    return (fixtures_dir / "sample_skill.md").read_text()


@pytest.fixture
def fake_config() -> "Config":  # noqa: F821
    from dr_agents_tester.config import Config

    return Config(
        api_key="test-token",
        endpoint="https://app.datarobot.com/api/v2",
        model="datarobot/test-model",
        test_model="datarobot/test-model",
    )

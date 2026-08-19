"""Tests for the sandbox env allowlist, fixture copying, and skill installation."""

import os
from pathlib import Path

import pytest

from dr_agents_tester.config import Config
from dr_agents_tester.eval.drivers import RunPaths
from dr_agents_tester.eval.models import BehavioralScenario, CheckSpec, Difficulty, FixtureSpec
from dr_agents_tester.eval.sandbox import build_agent_env, copy_fixtures, render_prompt
from dr_agents_tester.eval.skills_install import install_skills


def _scenario(source_dir: Path, **overrides: object) -> BehavioralScenario:
    defaults: dict[str, object] = dict(
        id="s",
        name="s",
        difficulty=Difficulty.EASY,
        prompt="Do it with {run_id}",
        skills_under_test=["some-skill"],
        success_checks=[CheckSpec(type="file_exists", params={"path": "x"})],
        source_dir=source_dir,
    )
    defaults.update(overrides)
    return BehavioralScenario(**defaults)  # type: ignore[arg-type]


class TestBuildAgentEnv:
    def test_allowlist_only(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path).create()
        cfg = Config(api_key="sekret", endpoint="https://app.datarobot.com/api/v2")
        env = build_agent_env(paths, cfg, "run-1")

        assert env["DATAROBOT_API_TOKEN"] == "sekret"
        assert env["HOME"] == str(paths.home)
        assert env["XDG_CONFIG_HOME"] == str(paths.home / ".config")
        assert env["TMPDIR"] == str(paths.tmp)
        assert env["DRAT_RUN_ID"] == "run-1"
        assert env["PATH"] == os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        # Nothing beyond the allowlist leaks in
        assert set(env) == {
            "PATH",
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
            "XDG_STATE_HOME",
            "TMPDIR",
            "DATAROBOT_API_TOKEN",
            "DATAROBOT_ENDPOINT",
            "DRAT_RUN_ID",
            "NO_COLOR",
            "CI",
            "TERM",
        }


class TestCopyFixtures:
    def test_copies_to_dest(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "train.csv").write_text("a\n1\n")
        paths = RunPaths(run_dir=tmp_path / "run").create()
        scenario = _scenario(
            src_dir, fixtures=[FixtureSpec(source="train.csv", dest="data/train.csv")]
        )
        written = copy_fixtures(scenario, paths)
        assert written == ["data/train.csv"]
        assert (paths.workspace / "data" / "train.csv").read_text() == "a\n1\n"

    def test_missing_fixture_raises(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        paths = RunPaths(run_dir=tmp_path / "run").create()
        scenario = _scenario(src_dir, fixtures=[FixtureSpec(source="nope.csv", dest="nope.csv")])
        with pytest.raises(FileNotFoundError, match="fixture not found"):
            copy_fixtures(scenario, paths)

    @pytest.mark.parametrize("bad_dest", ["/etc/pwned", "../escape.csv"])
    def test_escaping_dest_rejected(self, tmp_path: Path, bad_dest: str) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "f.csv").write_text("x")
        paths = RunPaths(run_dir=tmp_path / "run").create()
        scenario = _scenario(src_dir, fixtures=[FixtureSpec(source="f.csv", dest=bad_dest)])
        with pytest.raises(ValueError, match="workspace-relative"):
            copy_fixtures(scenario, paths)


class TestRenderPrompt:
    def test_run_id_and_epilogue(self, tmp_path: Path) -> None:
        scenario = _scenario(
            tmp_path, env={"resource_prefix": "{run_id}", "automl_mode": "quick"}
        )
        prompt = render_prompt(scenario, "drat-x-r1")
        assert "Do it with drat-x-r1" in prompt
        assert "starts with the prefix 'drat-x-r1'" in prompt

    def test_no_epilogue_without_resource_prefix(self, tmp_path: Path) -> None:
        scenario = _scenario(tmp_path)
        prompt = render_prompt(scenario, "drat-x-r1")
        assert "prefix" not in prompt.lower()


class TestInstallSkills:
    def _skills_source(self, tmp_path: Path) -> Path:
        src = tmp_path / "skills"
        for name in ("datarobot-a", "datarobot-b"):
            (src / name).mkdir(parents=True)
            (src / name / "SKILL.md").write_text(f"---\nname: {name}\n---\nBody.")
        # A nested skill layout (sub-skill with its own SKILL.md)
        nested = src / "datarobot-nested" / "sub-skill"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: sub-skill\n---\nBody.")
        # A non-skill dir that must be skipped
        (src / "not-a-skill").mkdir()
        (src / "not-a-skill" / "README.md").write_text("no SKILL.md here")
        return src

    def test_installs_all_skill_dirs(self, tmp_path: Path) -> None:
        src = self._skills_source(tmp_path)
        paths = RunPaths(run_dir=tmp_path / "run").create()
        installed = install_skills(src, paths)

        names = [s.name for s in installed]
        assert names == ["datarobot-a", "datarobot-b", "datarobot-nested"]
        target = paths.home / ".config" / "opencode" / "skills"
        assert (target / "datarobot-a" / "SKILL.md").is_file()
        assert (target / "datarobot-nested" / "sub-skill" / "SKILL.md").is_file()
        assert not (target / "not-a-skill").exists()

    def test_digests_stable_and_content_sensitive(self, tmp_path: Path) -> None:
        src = self._skills_source(tmp_path)
        paths1 = RunPaths(run_dir=tmp_path / "run1").create()
        paths2 = RunPaths(run_dir=tmp_path / "run2").create()
        first = install_skills(src, paths1)
        second = install_skills(src, paths2)
        assert [s.digest for s in first] == [s.digest for s in second]

        (src / "datarobot-a" / "SKILL.md").write_text("changed")
        paths3 = RunPaths(run_dir=tmp_path / "run3").create()
        third = install_skills(src, paths3)
        assert third[0].digest != first[0].digest
        assert third[1].digest == first[1].digest

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        paths = RunPaths(run_dir=tmp_path / "run").create()
        with pytest.raises(FileNotFoundError):
            install_skills(tmp_path / "nope", paths)

    def test_source_without_skills_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        paths = RunPaths(run_dir=tmp_path / "run").create()
        with pytest.raises(ValueError, match="No skill directories"):
            install_skills(empty, paths)

"""Tests for CLI parser construction (no network, no execution)."""

import pytest

from dr_agents_tester.cli import build_parser


class TestEvalRunBehavioralParser:
    def test_defaults(self) -> None:
        args = build_parser().parse_args(
            ["eval", "run-behavioral", "--scenarios", "tests/behavioral"]
        )
        assert args.scenarios == "tests/behavioral"
        assert args.driver == "opencode"
        assert args.n_runs == 3
        assert args.work_dir == ".behavioral-runs"
        assert args.output_dir == "results"
        assert args.skip_no_skill is False
        assert args.keep_resources is False
        assert args.fail_under_pass_rate is None
        assert args.run_prefix is None
        assert args.conditions is None

    def test_full_flag_surface(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "run-behavioral",
                "--scenarios",
                "s",
                "--skills-main",
                "../main/skills",
                "--skills-pr",
                "./skills",
                "--skip-no-skill",
                "--conditions",
                "skill_pr",
                "--driver",
                "fake",
                "--n-runs",
                "1",
                "--difficulty",
                "medium",
                "--work-dir",
                "/tmp/w",
                "--run-prefix",
                "drat-gh1-1",
                "--keep-resources",
                "--fail-under-pass-rate",
                "0.5",
            ]
        )
        assert args.skills_main == "../main/skills"
        assert args.conditions == "skill_pr"
        assert args.fail_under_pass_rate == 0.5
        assert args.keep_resources is True

    def test_scenarios_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["eval", "run-behavioral"])


class TestEvalSweepParser:
    def test_defaults_are_dry_run(self) -> None:
        args = build_parser().parse_args(["eval", "sweep"])
        assert args.prefix == "drat-"
        assert args.older_than_hours == 24.0
        assert args.execute is False

    def test_execute_flag(self) -> None:
        args = build_parser().parse_args(
            ["eval", "sweep", "--prefix", "drat-gh1-", "--older-than-hours", "0", "--execute"]
        )
        assert args.execute is True
        assert args.older_than_hours == 0.0


class TestExistingEvalRunUnchanged:
    def test_plan_eval_flags_still_parse(self) -> None:
        args = build_parser().parse_args(
            ["eval", "run", "--scenarios", "s", "--repo-tree", "tree.txt"]
        )
        assert args.n_runs == 5
        assert args.repo_tree == "tree.txt"

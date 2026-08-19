"""Tests for the success-check registry, local checks, and DataRobot checks (faked SDK)."""

from pathlib import Path
from typing import Any

import pytest

from dr_agents_tester.config import Config
from dr_agents_tester.eval.checks import (
    CHECK_REGISTRY,
    CheckContext,
    OutcomeCheck,
    build_checks,
    known_check_types,
    register_check,
    render_params,
    run_checks,
)
from dr_agents_tester.eval.models import CheckResult, CheckSpec


@pytest.fixture
def ctx(tmp_path: Path) -> CheckContext:
    return CheckContext(
        workspace=tmp_path,
        run_id="drat-test-abc123",
        config=Config(api_key="test-token", endpoint="https://app.datarobot.com/api/v2"),
    )


class TestRegistry:
    def test_v1_check_types_registered(self) -> None:
        assert {
            "file_exists",
            "dr_project_exists",
            "dr_deployment_healthy",
            "dr_predictions_returned",
        } <= known_check_types()

    def test_registry_importable_without_datarobot_sdk(self) -> None:
        # The registry (used by offline scenario validation) must not require
        # the optional SDK; this test suite itself runs without it installed.
        assert len(CHECK_REGISTRY) >= 4

    def test_collision_rejected(self) -> None:
        class Colliding(OutcomeCheck):
            type_name = "file_exists"

            def run(self, ctx: CheckContext) -> CheckResult:  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(ValueError, match="already registered"):
            register_check(Colliding)

    def test_build_checks_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown check type"):
            build_checks([CheckSpec(type="nope")], run_id="r1")


class TestRenderParams:
    def test_substitutes_run_id(self) -> None:
        params = {"name_contains": "prefix-{run_id}-suffix", "min_rows": 5}
        rendered = render_params(params, "abc")
        assert rendered == {"name_contains": "prefix-abc-suffix", "min_rows": 5}

    def test_other_braces_survive(self) -> None:
        params = {"pattern": "{not_a_template} {run_id}"}
        assert render_params(params, "x") == {"pattern": "{not_a_template} x"}


class TestFileExists:
    def test_pass(self, ctx: CheckContext) -> None:
        (ctx.workspace / "predictions.csv").write_text("a,b\n1,2\n")
        result = run_checks([CheckSpec(type="file_exists", params={"path": "predictions.csv"})], ctx)
        assert result.outcome_pass
        assert "predictions.csv" in result.checks[0].evidence

    def test_missing_file_fails(self, ctx: CheckContext) -> None:
        result = run_checks([CheckSpec(type="file_exists", params={"path": "nope.csv"})], ctx)
        assert not result.outcome_pass

    def test_min_bytes(self, ctx: CheckContext) -> None:
        (ctx.workspace / "tiny.txt").write_text("x")
        spec = CheckSpec(type="file_exists", params={"path": "tiny.txt", "min_bytes": 100})
        assert not run_checks([spec], ctx).outcome_pass

    def test_glob(self, ctx: CheckContext) -> None:
        (ctx.workspace / "out").mkdir()
        (ctx.workspace / "out" / "result_1.csv").write_text("data\n1\n")
        spec = CheckSpec(type="file_exists", params={"path": "out/*.csv"})
        assert run_checks([spec], ctx).outcome_pass

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.txt", "a/../../b"])
    def test_rejects_escaping_paths(self, ctx: CheckContext, bad: str) -> None:
        result = run_checks([CheckSpec(type="file_exists", params={"path": bad})], ctx)
        assert not result.outcome_pass
        assert result.checks[0].error is not None


class _FakeProject:
    def __init__(self, name: str, pid: str = "proj-1") -> None:
        self.project_name = name
        self.id = pid


class _FakeDeployment:
    def __init__(self, label: str, model: Any, health: str | None) -> None:
        self.label = label
        self.id = "dep-1"
        self.model = model
        self.service_health = {"status": health} if health else None


class _FakeDR:
    """Duck-typed stand-in for the datarobot module."""

    def __init__(self, projects: list[Any] = [], deployments: list[Any] = []) -> None:
        fake = self

        class Project:
            @staticmethod
            def list(search_params: dict[str, str] | None = None) -> list[Any]:
                needle = (search_params or {}).get("project_name", "")
                return [p for p in fake._projects if needle in p.project_name]

        class Deployment:
            @staticmethod
            def list(search: str | None = None) -> list[Any]:
                return [d for d in fake._deployments if (search or "") in d.label]

        class BatchPredictionJob:
            @staticmethod
            def list_by_status(statuses: list[str]) -> list[Any]:
                return []

        self._projects = projects
        self._deployments = deployments
        self.Project = Project
        self.Deployment = Deployment
        self.BatchPredictionJob = BatchPredictionJob


def _ctx_with_dr(tmp_path: Path, fake: _FakeDR) -> CheckContext:
    return CheckContext(
        workspace=tmp_path,
        run_id="drat-run-1",
        config=Config(api_key="t", endpoint="https://example.org/api/v2"),
        dr_client_factory=lambda: fake,
    )


class TestDataRobotChecks:
    def test_project_exists_pass(self, tmp_path: Path) -> None:
        fake = _FakeDR(projects=[_FakeProject("drat-run-1 churn model")])
        spec = CheckSpec(type="dr_project_exists", params={"name_contains": "{run_id}"})
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass
        assert "proj-1" in result.checks[0].evidence

    def test_project_exists_fail(self, tmp_path: Path) -> None:
        fake = _FakeDR(projects=[])
        spec = CheckSpec(type="dr_project_exists", params={"name_contains": "{run_id}"})
        assert not run_checks([spec], _ctx_with_dr(tmp_path, fake)).outcome_pass

    def test_deployment_healthy_pass_with_unknown_health(self, tmp_path: Path) -> None:
        fake = _FakeDR(
            deployments=[_FakeDeployment("drat-run-1 deploy", {"id": "m1"}, "unknown")]
        )
        spec = CheckSpec(type="dr_deployment_healthy", params={"name_contains": "{run_id}"})
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass

    def test_deployment_failing_health_fails(self, tmp_path: Path) -> None:
        fake = _FakeDR(
            deployments=[_FakeDeployment("drat-run-1 deploy", {"id": "m1"}, "failing")]
        )
        spec = CheckSpec(type="dr_deployment_healthy", params={"name_contains": "{run_id}"})
        assert not run_checks([spec], _ctx_with_dr(tmp_path, fake)).outcome_pass

    def test_deployment_without_model_fails(self, tmp_path: Path) -> None:
        fake = _FakeDR(deployments=[_FakeDeployment("drat-run-1 deploy", None, "passing")])
        spec = CheckSpec(type="dr_deployment_healthy", params={"name_contains": "{run_id}"})
        assert not run_checks([spec], _ctx_with_dr(tmp_path, fake)).outcome_pass

    def test_predictions_returned_from_csv(self, tmp_path: Path) -> None:
        rows = "\n".join(f"{i},0.{i}" for i in range(60))
        (tmp_path / "predictions.csv").write_text("row_id,churn_PREDICTION\n" + rows + "\n")
        fake = _FakeDR()
        spec = CheckSpec(type="dr_predictions_returned", params={"min_rows": 50})
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass
        assert "60 rows" in result.checks[0].evidence

    def test_predictions_too_few_rows(self, tmp_path: Path) -> None:
        (tmp_path / "predictions.csv").write_text("id,prediction\n1,0.5\n")
        spec = CheckSpec(type="dr_predictions_returned", params={"min_rows": 50})
        assert not run_checks([spec], _ctx_with_dr(tmp_path, _FakeDR())).outcome_pass

    def test_predictions_no_prediction_column(self, tmp_path: Path) -> None:
        (tmp_path / "predictions.csv").write_text("a,b\n1,2\n")
        spec = CheckSpec(type="dr_predictions_returned", params={})
        assert not run_checks([spec], _ctx_with_dr(tmp_path, _FakeDR())).outcome_pass

    def test_predictions_explicit_column(self, tmp_path: Path) -> None:
        (tmp_path / "out.csv").write_text("score\n0.1\n0.9\n")
        spec = CheckSpec(
            type="dr_predictions_returned",
            params={"path": "out.csv", "prediction_column": "score", "min_rows": 2},
        )
        assert run_checks([spec], _ctx_with_dr(tmp_path, _FakeDR())).outcome_pass


class TestRunChecks:
    def test_all_checks_run_despite_failure(self, ctx: CheckContext) -> None:
        (ctx.workspace / "second.txt").write_text("data")
        result = run_checks(
            [
                CheckSpec(type="file_exists", params={"path": "missing.txt"}),
                CheckSpec(type="file_exists", params={"path": "second.txt"}),
            ],
            ctx,
        )
        assert not result.outcome_pass
        assert [c.passed for c in result.checks] == [False, True]

    def test_exception_becomes_failed_result(self, tmp_path: Path) -> None:
        def boom() -> Any:
            raise RuntimeError("gateway down")

        ctx = CheckContext(
            workspace=tmp_path,
            run_id="r1",
            config=Config(api_key="t"),
            dr_client_factory=boom,
        )
        result = run_checks(
            [CheckSpec(type="dr_project_exists", params={"name_contains": "x"})], ctx
        )
        assert not result.outcome_pass
        assert "RuntimeError" in (result.checks[0].error or "")

    def test_run_id_templated_into_params(self, tmp_path: Path) -> None:
        captured: list[dict[str, object]] = []

        fake = _FakeDR()
        orig_list = fake.Project.list

        class RecordingProject:
            @staticmethod
            def list(search_params: dict[str, str] | None = None) -> list[Any]:
                captured.append(dict(search_params or {}))
                return orig_list(search_params)

        fake.Project = RecordingProject
        ctx = _ctx_with_dr(tmp_path, fake)
        run_checks([CheckSpec(type="dr_project_exists", params={"name_contains": "{run_id}"})], ctx)
        assert captured == [{"project_name": "drat-run-1"}]

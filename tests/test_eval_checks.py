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

    def test_substitutes_env_tokens(self) -> None:
        params = {"name_contains": "{env:BEHAVIORAL_FIXTURE_DEPLOYMENT_ID}"}
        rendered = render_params(params, "r1", {"BEHAVIORAL_FIXTURE_DEPLOYMENT_ID": "dep-42"})
        assert rendered == {"name_contains": "dep-42"}

    def test_missing_env_token_raises(self) -> None:
        params = {"name_contains": "{env:BEHAVIORAL_FIXTURE_DEPLOYMENT_ID}"}
        with pytest.raises(ValueError, match="BEHAVIORAL_FIXTURE_DEPLOYMENT_ID"):
            render_params(params, "r1", {})


class TestFileExists:
    def test_pass(self, ctx: CheckContext) -> None:
        (ctx.workspace / "predictions.csv").write_text("a,b\n1,2\n")
        result = run_checks(
            [CheckSpec(type="file_exists", params={"path": "predictions.csv"})], ctx
        )
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


class TestFileMatches:
    def test_pass(self, ctx: CheckContext) -> None:
        (ctx.workspace / "report.md").write_text("# Drift\n\nfeature drift is high\n")
        spec = CheckSpec(
            type="file_matches", params={"path": "report.md", "pattern": "feature drift"}
        )
        result = run_checks([spec], ctx)
        assert result.outcome_pass
        assert "1 match(es)" in result.checks[0].evidence

    def test_min_count(self, ctx: CheckContext) -> None:
        (ctx.workspace / "f.txt").write_text("tenure\ncharges\n")
        spec = CheckSpec(
            type="file_matches",
            params={"path": "f.txt", "pattern": "tenure|charges", "min_count": 3},
        )
        result = run_checks([spec], ctx)
        assert not result.outcome_pass
        assert "got f.txt: 2" in result.checks[0].evidence

    def test_ignorecase(self, ctx: CheckContext) -> None:
        (ctx.workspace / "f.txt").write_text("Feature Drift detected\n")
        spec = CheckSpec(
            type="file_matches",
            params={"path": "f.txt", "pattern": "feature drift", "ignorecase": True},
        )
        assert run_checks([spec], ctx).outcome_pass

    def test_glob_passes_when_any_file_matches(self, ctx: CheckContext) -> None:
        (ctx.workspace / "a_template.csv").write_text("no signal here\n")
        (ctx.workspace / "b_template.csv").write_text("tenure_months,charges\n0.0,1.0\n")
        spec = CheckSpec(
            type="file_matches", params={"path": "*template*.csv", "pattern": "tenure_months"}
        )
        assert run_checks([spec], ctx).outcome_pass

    def test_comment_prefixed_csv_matches_raw_text(self, ctx: CheckContext) -> None:
        # The predictions template writes a '#' metadata block before the
        # header — regex-on-raw-text must not care.
        (ctx.workspace / "prediction_template.csv").write_text(
            "# Deployment: dep-1\n# Target: churn\ntenure_months,contract_type\n0.0,sample_category\n"
        )
        spec = CheckSpec(
            type="file_matches",
            params={"path": "prediction_template.csv", "pattern": "^tenure_months,"},
        )
        assert run_checks([spec], ctx).outcome_pass

    def test_missing_file_fails(self, ctx: CheckContext) -> None:
        spec = CheckSpec(type="file_matches", params={"path": "nope.md", "pattern": "x"})
        result = run_checks([spec], ctx)
        assert not result.outcome_pass
        assert "no file matching" in result.checks[0].evidence

    def test_invalid_regex_is_error(self, ctx: CheckContext) -> None:
        (ctx.workspace / "f.txt").write_text("data")
        spec = CheckSpec(type="file_matches", params={"path": "f.txt", "pattern": "["})
        result = run_checks([spec], ctx)
        assert not result.outcome_pass
        assert "invalid regex" in (result.checks[0].error or "")

    def test_missing_pattern_is_error(self, ctx: CheckContext) -> None:
        spec = CheckSpec(type="file_matches", params={"path": "f.txt"})
        result = run_checks([spec], ctx)
        assert not result.outcome_pass
        assert "pattern" in (result.checks[0].error or "")

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.txt"])
    def test_rejects_escaping_paths(self, ctx: CheckContext, bad: str) -> None:
        spec = CheckSpec(type="file_matches", params={"path": bad, "pattern": "x"})
        result = run_checks([spec], ctx)
        assert not result.outcome_pass
        assert result.checks[0].error is not None

    def test_oversized_file_skipped(self, ctx: CheckContext) -> None:
        (ctx.workspace / "big.txt").write_text("needle " * 10)
        spec = CheckSpec(
            type="file_matches", params={"path": "big.txt", "pattern": "needle", "max_bytes": 10}
        )
        result = run_checks([spec], ctx)
        assert not result.outcome_pass
        assert "max_bytes" in (result.checks[0].error or "")

    def test_undecodable_bytes_never_crash(self, ctx: CheckContext) -> None:
        (ctx.workspace / "bin.dat").write_bytes(b"\xff\xfe needle \xff")
        spec = CheckSpec(type="file_matches", params={"path": "bin.dat", "pattern": "needle"})
        assert run_checks([spec], ctx).outcome_pass


class _FakeProject:
    def __init__(self, name: str, pid: str = "proj-1", stage: str = "modeling") -> None:
        self.project_name = name
        self.id = pid
        self.stage = stage


class _FakeUseCase:
    def __init__(self, name: str, ucid: str = "uc-1") -> None:
        self.name = name
        self.id = ucid


class _FakeDeployment:
    def __init__(self, label: str, model: Any, health: str | None) -> None:
        self.label = label
        self.id = "dep-1"
        self.model = model
        self.service_health = {"status": health} if health else None


class _FakeDR:
    """Duck-typed stand-in for the datarobot module."""

    def __init__(
        self,
        projects: list[Any] = [],
        deployments: list[Any] = [],
        use_cases: list[Any] = [],
        rest_responses: dict[str, Any] | None = None,
    ) -> None:
        fake = self

        class Project:
            @staticmethod
            def list(search_params: dict[str, str] | None = None) -> list[Any]:
                needle = (search_params or {}).get("project_name", "")
                return [p for p in fake._projects if needle in p.project_name]

            @staticmethod
            def get(pid: str) -> Any:
                return next(p for p in fake._projects if p.id == pid)

        class Deployment:
            @staticmethod
            def list(search: str | None = None) -> list[Any]:
                return [d for d in fake._deployments if (search or "") in d.label]

        class BatchPredictionJob:
            @staticmethod
            def list_by_status(statuses: list[str]) -> list[Any]:
                return []

        class UseCase:
            @staticmethod
            def list() -> list[Any]:
                return list(fake._use_cases)

        class _RestResponse:
            def __init__(self, payload: Any) -> None:
                self._payload = payload

            def json(self) -> Any:
                return self._payload

        class _RestClient:
            def get(self, path: str) -> Any:
                fake.rest_calls.append(path)
                return _RestResponse(fake._rest_responses[path])

        class client:
            @staticmethod
            def get_client() -> Any:
                return _RestClient()

        self._projects = projects
        self._deployments = deployments
        self._use_cases = use_cases
        self._rest_responses = rest_responses or {}
        self.rest_calls: list[str] = []
        self.Project = Project
        self.Deployment = Deployment
        self.BatchPredictionJob = BatchPredictionJob
        self.UseCase = UseCase
        self.client = client


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
        fake = _FakeDR(deployments=[_FakeDeployment("drat-run-1 deploy", {"id": "m1"}, "unknown")])
        spec = CheckSpec(type="dr_deployment_healthy", params={"name_contains": "{run_id}"})
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass

    def test_deployment_failing_health_fails(self, tmp_path: Path) -> None:
        fake = _FakeDR(deployments=[_FakeDeployment("drat-run-1 deploy", {"id": "m1"}, "failing")])
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


class TestDrUseCaseExists:
    def test_pass(self, tmp_path: Path) -> None:
        fake = _FakeDR(use_cases=[_FakeUseCase("drat-run-1 churn use case")])
        spec = CheckSpec(type="dr_use_case_exists", params={"name_contains": "{run_id}"})
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass
        assert "uc-1" in result.checks[0].evidence

    def test_fail(self, tmp_path: Path) -> None:
        fake = _FakeDR(use_cases=[_FakeUseCase("unrelated")])
        spec = CheckSpec(type="dr_use_case_exists", params={"name_contains": "{run_id}"})
        assert not run_checks([spec], _ctx_with_dr(tmp_path, fake)).outcome_pass

    def test_defaults_to_run_id(self, tmp_path: Path) -> None:
        fake = _FakeDR(use_cases=[_FakeUseCase("drat-run-1")])
        assert run_checks(
            [CheckSpec(type="dr_use_case_exists")], _ctx_with_dr(tmp_path, fake)
        ).outcome_pass


class TestDrTracesReceived:
    def _fake(self, count: int, ucid: str = "uc-1") -> _FakeDR:
        payload = {
            "count": count,
            "data": [{"traceId": "abc123", "spansCount": 5}] if count else [],
        }
        return _FakeDR(
            use_cases=[_FakeUseCase("drat-run-1 otel use case", ucid=ucid)],
            rest_responses={f"otel/experiment_container/{ucid}/traces/": payload},
        )

    def test_pass(self, tmp_path: Path) -> None:
        fake = self._fake(count=2)
        spec = CheckSpec(
            type="dr_traces_received",
            params={"use_case_name_contains": "{run_id}", "deadline_seconds": 0},
        )
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass
        assert "traceId=abc123" in result.checks[0].evidence
        assert fake.rest_calls == ["otel/experiment_container/uc-1/traces/"]

    def test_no_traces_fails_after_deadline(self, tmp_path: Path) -> None:
        fake = self._fake(count=0)
        spec = CheckSpec(
            type="dr_traces_received",
            params={"use_case_name_contains": "{run_id}", "deadline_seconds": 0},
        )
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert not result.outcome_pass
        assert "0 trace(s)" in result.checks[0].evidence

    def test_missing_use_case_fails(self, tmp_path: Path) -> None:
        fake = _FakeDR(use_cases=[])
        spec = CheckSpec(type="dr_traces_received", params={"deadline_seconds": 0})
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert not result.outcome_pass
        assert "no use case" in result.checks[0].evidence

    def test_min_traces_threshold(self, tmp_path: Path) -> None:
        fake = self._fake(count=1)
        spec = CheckSpec(
            type="dr_traces_received",
            params={
                "use_case_name_contains": "{run_id}",
                "min_traces": 3,
                "deadline_seconds": 0,
            },
        )
        assert not run_checks([spec], _ctx_with_dr(tmp_path, fake)).outcome_pass


class TestDrProjectStage:
    def test_stage_pass(self, tmp_path: Path) -> None:
        fake = _FakeDR(projects=[_FakeProject("drat-run-1 churn", stage="modeling")])
        spec = CheckSpec(
            type="dr_project_exists", params={"name_contains": "{run_id}", "stage": "modeling"}
        )
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass
        assert "reached stage 'modeling'" in result.checks[0].evidence

    def test_wrong_stage_fails_single_shot(self, tmp_path: Path) -> None:
        fake = _FakeDR(projects=[_FakeProject("drat-run-1 churn", stage="eda")])
        spec = CheckSpec(
            type="dr_project_exists", params={"name_contains": "{run_id}", "stage": "modeling"}
        )
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert not result.outcome_pass
        assert "'eda'" in result.checks[0].evidence

    def test_polling_picks_up_late_stage(self, tmp_path: Path) -> None:
        project = _FakeProject("drat-run-1 churn", stage="eda")
        fake = _FakeDR(projects=[project])
        lists = 0
        orig_list = fake.Project.list

        class AdvancingProject:
            @staticmethod
            def list(search_params: dict[str, str] | None = None) -> list[Any]:
                nonlocal lists
                lists += 1
                if lists >= 2:
                    project.stage = "modeling"
                return orig_list(search_params)

            get = fake.Project.get

        fake.Project = AdvancingProject
        spec = CheckSpec(
            type="dr_project_exists",
            params={
                "name_contains": "{run_id}",
                "stage": "modeling",
                "deadline_seconds": 30,
                "poll_seconds": 0,
            },
        )
        result = run_checks([spec], _ctx_with_dr(tmp_path, fake))
        assert result.outcome_pass
        assert lists == 2

    def test_no_stage_param_keeps_v1_behavior(self, tmp_path: Path) -> None:
        fake = _FakeDR(projects=[_FakeProject("drat-run-1 churn", stage="eda")])
        spec = CheckSpec(type="dr_project_exists", params={"name_contains": "{run_id}"})
        assert run_checks([spec], _ctx_with_dr(tmp_path, fake)).outcome_pass


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

    def test_host_env_templated_into_params(self, tmp_path: Path) -> None:
        fake = _FakeDR(deployments=[_FakeDeployment("fixture dep-42", {"id": "m1"}, "passing")])
        ctx = _ctx_with_dr(tmp_path, fake)
        spec = CheckSpec(
            type="dr_deployment_healthy",
            params={"name_contains": "{env:BEHAVIORAL_FIXTURE_DEPLOYMENT_ID}"},
        )
        result = run_checks([spec], ctx, host_env={"BEHAVIORAL_FIXTURE_DEPLOYMENT_ID": "dep-42"})
        assert result.outcome_pass

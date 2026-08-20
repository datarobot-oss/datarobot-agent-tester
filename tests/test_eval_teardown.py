"""Tests for resource teardown and the sweeper, against a recording fake SDK."""

import datetime as dt
from typing import Any

from dr_agents_tester.eval.checks.teardown import TeardownReport, delete_run_resources


class _Resource:
    def __init__(self, name_attr: str, name: str, created: dt.datetime | None = None) -> None:
        setattr(self, name_attr, name)
        self.id = f"id-{name}"
        if created is not None:
            self.created = created
        self.deleted = False

    def delete(self) -> None:
        self.deleted = True


class _FakeDR:
    def __init__(
        self,
        deployments: list[Any] = [],
        projects: list[Any] = [],
        datasets: list[Any] = [],
        use_cases: list[Any] = [],
        fail_on: set[str] = set(),
    ) -> None:
        fake = self
        self._fail_on = set(fail_on)
        self.deleted_by_classmethod: list[str] = []

        class Deployment:
            @staticmethod
            def list(search: str | None = None) -> list[Any]:
                return list(deployments)

        class Project:
            @staticmethod
            def list(search_params: dict[str, str] | None = None) -> list[Any]:
                return list(projects)

        class Dataset:
            @staticmethod
            def list() -> list[Any]:
                return list(datasets)

            @staticmethod
            def delete(dataset_id: str) -> None:
                if dataset_id in fake._fail_on:
                    raise RuntimeError("cannot delete")
                fake.deleted_by_classmethod.append(dataset_id)

        class UseCase:
            @staticmethod
            def list() -> list[Any]:
                return list(use_cases)

            @staticmethod
            def delete(use_case_id: str) -> None:
                fake.deleted_by_classmethod.append(use_case_id)

        self.Deployment = Deployment
        self.Project = Project
        self.Dataset = Dataset
        self.UseCase = UseCase


class TestDeleteRunResources:
    def test_deletes_only_prefixed(self) -> None:
        mine = _Resource("label", "drat-run-1 deployment")
        other = _Resource("label", "production deployment")
        fake = _FakeDR(deployments=[mine, other])

        report = delete_run_resources(fake, "drat-run-1")

        assert mine.deleted and not other.deleted
        assert report.deleted["deployments"] == ["drat-run-1 deployment"]

    def test_children_before_parents_order(self) -> None:
        order: list[str] = []

        class Tracking(_Resource):
            def delete(self) -> None:
                order.append(type(self).kind)  # type: ignore[attr-defined]
                super().delete()

        class Dep(Tracking):
            kind = "deployment"

        class Proj(Tracking):
            kind = "project"

        fake = _FakeDR(
            deployments=[Dep("label", "drat-x d")],
            projects=[Proj("project_name", "drat-x p")],
        )
        delete_run_resources(fake, "drat-x")
        assert order == ["deployment", "project"]

    def test_failure_isolated_and_reported(self) -> None:
        ds1 = _Resource("name", "drat-x ds1")
        ds1.id = "ds-fail"
        ds2 = _Resource("name", "drat-x ds2")
        fake = _FakeDR(datasets=[ds1, ds2], fail_on={"ds-fail"})

        report = delete_run_resources(fake, "drat-x")

        assert fake.deleted_by_classmethod == [ds2.id]
        assert len(report.failed["datasets"]) == 1
        assert "RuntimeError" in report.failed["datasets"][0]

    def test_dry_run_deletes_nothing(self) -> None:
        dep = _Resource("label", "drat-x d")
        fake = _FakeDR(deployments=[dep])
        report = delete_run_resources(fake, "drat-x", dry_run=True)
        assert not dep.deleted
        assert report.deleted["deployments"] == ["drat-x d"]
        assert report.dry_run


class TestAgeThreshold:
    def _old(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)

    def test_old_resources_swept(self) -> None:
        dep = _Resource("label", "drat-x d", created=self._old())
        fake = _FakeDR(deployments=[dep])
        delete_run_resources(fake, "drat-", older_than_hours=24)
        assert dep.deleted

    def test_young_resources_kept(self) -> None:
        dep = _Resource("label", "drat-x d", created=dt.datetime.now(dt.timezone.utc))
        fake = _FakeDR(deployments=[dep])
        delete_run_resources(fake, "drat-", older_than_hours=24)
        assert not dep.deleted

    def test_age_from_name_stamp(self) -> None:
        # No SDK timestamp; the make_run_id-style stamp in the name decides.
        old_stamp = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)).strftime(
            "%Y%m%d%H%M%S"
        )
        dep = _Resource("label", f"drat-x-no-skill-r1-{old_stamp}abcd deployment")
        fake = _FakeDR(deployments=[dep])
        delete_run_resources(fake, "drat-", older_than_hours=24)
        assert dep.deleted

    def test_unknown_age_skipped(self) -> None:
        dep = _Resource("label", "drat-mystery deployment")  # no timestamp anywhere
        fake = _FakeDR(deployments=[dep])
        report = delete_run_resources(fake, "drat-", older_than_hours=24)
        assert not dep.deleted
        assert report.deleted == {}

    def test_zero_threshold_ignores_age(self) -> None:
        dep = _Resource("label", "drat-mystery deployment")
        fake = _FakeDR(deployments=[dep])
        delete_run_resources(fake, "drat-", older_than_hours=0)
        assert dep.deleted


class TestTeardownReport:
    def test_record_helpers(self) -> None:
        report = TeardownReport()
        report.record("projects", "drat-p")
        report.record_failure("datasets", "drat-d", RuntimeError("x"))
        assert report.deleted == {"projects": ["drat-p"]}
        assert report.failed["datasets"] == ["drat-d: RuntimeError: x"]

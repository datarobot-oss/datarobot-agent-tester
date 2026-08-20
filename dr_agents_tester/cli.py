"""CLI entry point for datarobot-agent-tester.

Commands
--------
dr-agent agents generate   Generate/update AGENTS.md
dr-agent agents test       Evaluate AGENTS.md
dr-agent agents revise     Revise AGENTS.md from saved report

dr-agent skills test       Evaluate a skill file (or all with --all)
dr-agent skills improve    Improve a skill from its saved report

dr-agent eval run             Run AGENTS.md evaluation framework
dr-agent eval run-behavioral  Run skill behavioral evaluation (agent in sandbox)
dr-agent eval report          Regenerate report from saved results
dr-agent eval sweep           Delete leaked DataRobot resources by run prefix
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from .config import Config

_has_dotenv = importlib.util.find_spec("dotenv") is not None


def _load_env() -> None:
    if _has_dotenv:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"), override=False)


def _make_config(args: argparse.Namespace) -> Config:
    # Config() reads all env vars and applies defaults; only override if
    # the caller explicitly passed --model / --test-model flags.
    cfg = Config()
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "test_model", None):
        cfg.test_model = args.test_model
    return cfg


# ---------------------------------------------------------------------------
# agents sub-commands
# ---------------------------------------------------------------------------


def cmd_agents_generate(args: argparse.Namespace) -> None:
    from .agents_md import AgentsMd
    from .context import find_repo_root

    cfg = _make_config(args)
    cfg.validate()
    cwd = Path.cwd().resolve()
    target_dir = (cwd / args.dir).resolve()
    repo_root = find_repo_root(target_dir)
    AgentsMd(cfg).generate(
        target_dir,
        repo_root,
        dry_run=args.dry_run,
        no_copilot=args.no_copilot,
        model=args.model or None,
    )


def cmd_agents_test(args: argparse.Namespace) -> None:
    from .agents_md import AgentsMd
    from .context import find_repo_root

    cfg = _make_config(args)
    cfg.validate()
    cwd = Path.cwd().resolve()
    target_dir = (cwd / args.dir).resolve()
    repo_root = find_repo_root(target_dir)
    AgentsMd(cfg).test(target_dir, repo_root, test_model=args.test_model or None)


def cmd_agents_revise(args: argparse.Namespace) -> None:
    from .agents_md import AgentsMd
    from .context import find_repo_root

    cfg = _make_config(args)
    cfg.validate()
    cwd = Path.cwd().resolve()
    target_dir = (cwd / args.dir).resolve()
    repo_root = find_repo_root(target_dir)
    AgentsMd(cfg).revise(
        target_dir,
        repo_root,
        dry_run=args.dry_run,
        no_copilot=args.no_copilot,
        model=args.model or None,
    )


# ---------------------------------------------------------------------------
# skills sub-commands
# ---------------------------------------------------------------------------


def cmd_skills_test(args: argparse.Namespace) -> None:
    from .skills import Skills

    cfg = _make_config(args)
    cfg.validate()
    skills = Skills(cfg)

    if args.all:
        skills_dir = Path(args.dir).resolve()
        skills.test_all(skills_dir, test_model=args.test_model or None)
    else:
        if not args.skill:
            print("❌ Provide --skill <path> or use --all", file=sys.stderr)
            sys.exit(1)
        skills.test(Path(args.skill), test_model=args.test_model or None)


def cmd_skills_improve(args: argparse.Namespace) -> None:
    from .skills import Skills

    cfg = _make_config(args)
    cfg.validate()
    if not args.skill:
        print("❌ Provide --skill <path>", file=sys.stderr)
        sys.exit(1)
    Skills(cfg).improve(Path(args.skill), dry_run=args.dry_run, model=args.model or None)


# ---------------------------------------------------------------------------
# eval sub-commands
# ---------------------------------------------------------------------------


def cmd_eval_run(args: argparse.Namespace) -> None:
    from .eval.models import ConditionType, Difficulty, EvalCondition
    from .eval.runner import Evaluator

    cfg = _make_config(args)
    cfg.validate()

    scenarios_dir = Path(args.scenarios).resolve()
    repo_tree_path = Path(args.repo_tree).resolve()

    conditions: list[EvalCondition] = [EvalCondition(condition_type=ConditionType.NO_CONTEXT)]

    if args.agents_generated:
        content = Path(args.agents_generated).read_text()
        conditions.append(
            EvalCondition(condition_type=ConditionType.GENERATED, agents_md_content=content)
        )

    if args.agents_refined:
        content = Path(args.agents_refined).read_text()
        conditions.append(
            EvalCondition(condition_type=ConditionType.REFINED, agents_md_content=content)
        )

    difficulty_filter = None
    if args.difficulty:
        difficulty_filter = Difficulty(args.difficulty)

    evaluator = Evaluator(
        config=cfg,
        scenarios_dir=scenarios_dir,
        repo_tree_path=repo_tree_path,
        conditions=conditions,
        n_runs=args.n_runs,
        difficulty_filter=difficulty_filter,
    )

    print(f"Running evaluation: {len(conditions)} conditions, n_runs={args.n_runs}")
    report = evaluator.run()

    output_dir = Path(args.output_dir).resolve()
    md_path, json_path = evaluator.save_report(report, output_dir)
    print(f"\nReport saved to:\n  {md_path}\n  {json_path}")


def cmd_eval_run_behavioral(args: argparse.Namespace) -> None:
    from .eval.backends import AgentBackend
    from .eval.drivers import get_driver
    from .eval.models import ConditionType, Difficulty, EvalCondition
    from .eval.runner import Evaluator
    from .eval.stats import outcome_pass_rate

    cfg = _make_config(args)
    cfg.validate()

    conditions: list[EvalCondition] = []
    if not args.skip_no_skill:
        conditions.append(EvalCondition(condition_type=ConditionType.NO_SKILL))
    if args.skills_main:
        conditions.append(
            EvalCondition(
                condition_type=ConditionType.SKILL_MAIN,
                skills_source=Path(args.skills_main).resolve(),
            )
        )
    if args.skills_pr:
        conditions.append(
            EvalCondition(
                condition_type=ConditionType.SKILL_PR,
                skills_source=Path(args.skills_pr).resolve(),
            )
        )

    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
        unknown = wanted - {c.condition_type.value for c in conditions}
        if unknown:
            print(
                f"❌ --conditions names not in this run: {', '.join(sorted(unknown))}",
                file=sys.stderr,
            )
            sys.exit(2)
        conditions = [c for c in conditions if c.condition_type.value in wanted]

    if not conditions:
        print(
            "❌ No conditions to run. Provide --skills-main/--skills-pr or drop --skip-no-skill.",
            file=sys.stderr,
        )
        sys.exit(2)

    from .eval.checks.teardown import delete_run_resources_for_config

    backend = AgentBackend(
        config=cfg,
        driver=get_driver(args.driver),
        work_dir=Path(args.work_dir).resolve(),
        teardown=None if args.keep_resources else delete_run_resources_for_config,
        keep_resources=args.keep_resources,
    )

    difficulty_filter = Difficulty(args.difficulty) if args.difficulty else None

    evaluator = Evaluator(
        config=cfg,
        scenarios_dir=[Path(p).resolve() for p in args.scenarios],
        conditions=conditions,
        n_runs=args.n_runs,
        difficulty_filter=difficulty_filter,
        backend=backend,
        run_prefix=args.run_prefix,
    )

    print(
        f"Running behavioral evaluation: driver={args.driver}, "
        f"{len(conditions)} condition(s), n_runs={args.n_runs}"
    )
    try:
        report = evaluator.run()
    except ValueError as exc:
        # Pre-run validation (missing fixture env vars, duplicate scenario
        # ids) aborts before any agent tokens are spent.
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir).resolve()
    md_path, json_path = evaluator.save_report(report, output_dir)
    print(f"\nReport saved to:\n  {md_path}\n  {json_path}")

    if args.fail_under_pass_rate is not None:
        rates = [
            (ct.value, outcome_pass_rate(report, ct))
            for ct in (c.condition_type for c in conditions)
        ]
        failing = [
            (name, rate)
            for name, rate in rates
            if rate is not None and rate < args.fail_under_pass_rate
        ]
        if failing:
            for name, rate in failing:
                print(
                    f"❌ {name}: pass rate {rate:.0%} < gate {args.fail_under_pass_rate:.0%}",
                    file=sys.stderr,
                )
            sys.exit(1)


def cmd_eval_sweep(args: argparse.Namespace) -> None:
    from .eval.checks.teardown import sweep

    cfg = _make_config(args)
    cfg.validate()
    report = sweep(
        cfg,
        prefix=args.prefix,
        older_than_hours=args.older_than_hours,
        dry_run=not args.execute,
    )
    if report.failed:
        sys.exit(1)


def cmd_eval_report(args: argparse.Namespace) -> None:
    import json as json_mod

    from .eval.report import dict_to_report, generate_markdown_report

    results_path = Path(args.results).resolve()
    data = json_mod.loads(results_path.read_text())
    report = dict_to_report(data)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "eval_report.md"
    md_path.write_text(generate_markdown_report(report))
    print(f"Report regenerated: {md_path}")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="Generation model (overrides env/default)")
    p.add_argument("--test-model", default=None, help="Evaluation model (overrides env/default)")


def _add_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dir",
        default=".",
        help="Target directory. Default: current directory.",
    )


def _add_dry_run(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output without writing any files.",
    )


def _add_no_copilot(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-copilot",
        action="store_true",
        help="Skip writing .github/copilot-instructions.md.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr-agent",
        description="Generate, test, and improve AGENTS.md files and AI coding skills.",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # ---- agents ----
    agents_p = sub.add_parser("agents", help="AGENTS.md operations")
    agents_sub = agents_p.add_subparsers(dest="command", required=True)

    gen_p = agents_sub.add_parser("generate", help="Generate or update AGENTS.md")
    _add_dir_arg(gen_p)
    _add_dry_run(gen_p)
    _add_no_copilot(gen_p)
    _add_common_model_args(gen_p)
    gen_p.set_defaults(func=cmd_agents_generate)

    test_p = agents_sub.add_parser("test", help="Evaluate AGENTS.md on disk")
    _add_dir_arg(test_p)
    _add_common_model_args(test_p)
    test_p.set_defaults(func=cmd_agents_test)

    rev_p = agents_sub.add_parser("revise", help="Revise AGENTS.md from saved test report")
    _add_dir_arg(rev_p)
    _add_dry_run(rev_p)
    _add_no_copilot(rev_p)
    _add_common_model_args(rev_p)
    rev_p.set_defaults(func=cmd_agents_revise)

    # ---- skills ----
    skills_p = sub.add_parser("skills", help="AI coding skill operations")
    skills_sub = skills_p.add_subparsers(dest="command", required=True)

    sk_test_p = skills_sub.add_parser("test", help="Evaluate a skill file")
    sk_test_p.add_argument("--skill", default=None, help="Path to skill file")
    sk_test_p.add_argument("--all", action="store_true", help="Test all skills in --dir")
    sk_test_p.add_argument(
        "--dir", default="skills", help="Skills directory for --all. Default: skills/"
    )
    _add_common_model_args(sk_test_p)
    sk_test_p.set_defaults(func=cmd_skills_test)

    sk_imp_p = skills_sub.add_parser("improve", help="Improve a skill from its saved report")
    sk_imp_p.add_argument("--skill", default=None, help="Path to skill file")
    _add_dry_run(sk_imp_p)
    _add_common_model_args(sk_imp_p)
    sk_imp_p.set_defaults(func=cmd_skills_improve)

    # ---- eval ----
    eval_p = sub.add_parser("eval", help="AGENTS.md evaluation framework")
    eval_sub = eval_p.add_subparsers(dest="command", required=True)

    eval_run_p = eval_sub.add_parser("run", help="Run AGENTS.md evaluation")
    eval_run_p.add_argument(
        "--scenarios", required=True, help="Directory containing scenario YAML files"
    )
    eval_run_p.add_argument("--repo-tree", required=True, help="Path to repo file tree text file")
    eval_run_p.add_argument("--agents-generated", default=None, help="Path to generated AGENTS.md")
    eval_run_p.add_argument("--agents-refined", default=None, help="Path to refined AGENTS.md")
    eval_run_p.add_argument("--n-runs", type=int, default=5, help="Runs per scenario (default: 5)")
    eval_run_p.add_argument(
        "--difficulty",
        default=None,
        choices=["easy", "medium", "hard", "expert"],
        help="Only run scenarios of this difficulty",
    )
    eval_run_p.add_argument(
        "--output-dir", default="results", help="Output directory for reports (default: results/)"
    )
    _add_common_model_args(eval_run_p)
    eval_run_p.set_defaults(func=cmd_eval_run)

    eval_bhv_p = eval_sub.add_parser(
        "run-behavioral", help="Run skill behavioral evaluation (real agent in sandbox)"
    )
    eval_bhv_p.add_argument(
        "--scenarios",
        action="append",
        required=True,
        help=(
            "Directory containing behavioral scenario YAML files (repeatable; each "
            "directory is scanned along with one subdirectory level, so "
            "tests/behavioral/scenarios picks up every scenarios/<skill>/ dir)"
        ),
    )
    eval_bhv_p.add_argument(
        "--skills-main", default=None, help="Skills dir (main checkout) → skill_main condition"
    )
    eval_bhv_p.add_argument(
        "--skills-pr", default=None, help="Skills dir (PR checkout / working tree) → skill_pr"
    )
    eval_bhv_p.add_argument(
        "--skip-no-skill",
        action="store_true",
        help="Skip the no_skill baseline condition (included by default)",
    )
    eval_bhv_p.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated subset of conditions to run (for CI matrix jobs)",
    )
    eval_bhv_p.add_argument(
        "--driver", default="opencode", help="Agent driver name (default: opencode)"
    )
    eval_bhv_p.add_argument("--n-runs", type=int, default=3, help="Runs per cell (default: 3)")
    eval_bhv_p.add_argument(
        "--difficulty",
        default=None,
        choices=["easy", "medium", "hard", "expert"],
        help="Only run scenarios of this difficulty",
    )
    eval_bhv_p.add_argument(
        "--work-dir",
        default=".behavioral-runs",
        help="Directory for per-run workspaces and transcripts (default: .behavioral-runs/)",
    )
    eval_bhv_p.add_argument(
        "--output-dir", default="results", help="Output directory for reports (default: results/)"
    )
    eval_bhv_p.add_argument(
        "--run-prefix",
        default=None,
        help="Run-id prefix (CI passes e.g. drat-gh<run_id>-<attempt>; default: drat-<scenario>)",
    )
    eval_bhv_p.add_argument(
        "--keep-resources",
        action="store_true",
        help="Skip per-run DataRobot resource teardown (for debugging)",
    )
    eval_bhv_p.add_argument(
        "--fail-under-pass-rate",
        type=float,
        default=None,
        help="Exit non-zero when any condition's pass rate falls below this (absent = advisory)",
    )
    _add_common_model_args(eval_bhv_p)
    eval_bhv_p.set_defaults(func=cmd_eval_run_behavioral)

    eval_report_p = eval_sub.add_parser("report", help="Regenerate report from saved results")
    eval_report_p.add_argument("--results", required=True, help="Path to eval_results.json")
    eval_report_p.add_argument(
        "--output-dir", default="results", help="Output directory (default: results/)"
    )
    eval_report_p.set_defaults(func=cmd_eval_report)

    eval_sweep_p = eval_sub.add_parser(
        "sweep", help="Delete leaked DataRobot resources created by behavioral runs"
    )
    eval_sweep_p.add_argument(
        "--prefix", default="drat-", help="Resource-name prefix to match (default: drat-)"
    )
    eval_sweep_p.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Only delete resources older than this many hours (default: 24)",
    )
    eval_sweep_p.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete. Without this flag the sweep is a dry run.",
    )
    _add_common_model_args(eval_sweep_p)
    eval_sweep_p.set_defaults(func=cmd_eval_sweep)

    return parser


def main() -> None:
    _load_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

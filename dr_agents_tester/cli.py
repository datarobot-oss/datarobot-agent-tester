"""CLI entry point for datarobot-agent-tester.

Commands
--------
dr-agent agents generate   Generate/update AGENTS.md
dr-agent agents test       Evaluate AGENTS.md
dr-agent agents revise     Revise AGENTS.md from saved report

dr-agent skills test       Evaluate a skill file (or all with --all)
dr-agent skills improve    Improve a skill from its saved report

dr-agent eval run          Run AGENTS.md evaluation framework
dr-agent eval report       Regenerate report from saved results
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
# skillopt sub-commands
# ---------------------------------------------------------------------------


def cmd_skillopt_gen_rows(args: argparse.Namespace) -> None:
    from .skillopt.eval_gen import generate_rows, save_rows

    cfg = _make_config(args)
    cfg.validate()
    skill_text = Path(args.skill).read_text()
    docs = ""
    if args.docs:
        docs = Path(args.docs).read_text()
    rows = generate_rows(
        skill_text=skill_text,
        docs_context=docs,
        n=args.n,
        code_pct=args.code_pct,
        rubric_pct=100 - args.code_pct,
        model=args.model or cfg.model,
        config=cfg,
    )
    out = Path(args.out).resolve()
    save_rows(rows, out)
    print(f"✅ {len(rows)} rows -> {out}")


def cmd_skillopt_run(args: argparse.Namespace) -> None:
    from .skillopt.eval_gen import load_rows
    from .skillopt.loop import LoopConfig, SkillOptLoop
    from .skillopt.scorers import CompositeScorer, MockExecScorer, RubricScorer

    cfg = _make_config(args)
    cfg.validate()
    rows = load_rows(Path(args.rows).resolve())
    if args.limit:
        rows = rows[: args.limit]
    scorer = CompositeScorer(
        scorers=[
            MockExecScorer(),
            RubricScorer(config=cfg, judge_model=args.test_model or cfg.test_model),
        ]
    )
    loop = SkillOptLoop(
        skill_path=Path(args.skill).resolve(),
        rows=rows,
        scorer=scorer,
        config=cfg,
        loop_config=LoopConfig(
            iters=args.iters,
            rollout_model=args.test_model or cfg.test_model,
            optimizer_model=args.model or cfg.model,
            parallel_rollouts=args.parallel,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            drop_baseline_below=args.filter_baseline_below,
            lt_max=args.lt_max,
            lt_floor=args.lt_floor,
            lt_schedule=args.lt_schedule,
            n_candidates=args.n_candidates,
            max_edit_chars=args.max_edit_chars,
        ),
    )
    summary = loop.run()
    print(f"\n📊 baseline_test={summary['baseline_test']:.3f}  "
          f"final_test={summary['final_test']:.3f}  "
          f"Δ={summary['final_test'] - summary['baseline_test']:+.3f}")
    print(f"📁 results: {loop.run_dir}")
    print(f"🌐 open file://{loop.run_dir.resolve()}/report.html")


def cmd_skillopt_score(args: argparse.Namespace) -> None:
    """Score a skill against a rows file without optimizing — useful for baselines."""
    from .skillopt.eval_gen import load_rows
    from .skillopt.loop import LoopConfig, _score_set, _mean
    from .skillopt.scorers import CompositeScorer, MockExecScorer, RubricScorer

    cfg = _make_config(args)
    cfg.validate()
    rows = load_rows(Path(args.rows).resolve())
    if args.limit:
        rows = rows[: args.limit]
    scorer = CompositeScorer(
        scorers=[
            MockExecScorer(),
            RubricScorer(config=cfg, judge_model=args.test_model or cfg.test_model),
        ]
    )
    skill = Path(args.skill).read_text()
    scores = _score_set(
        skill,
        rows,
        scorer,
        LoopConfig(
            rollout_model=args.test_model or cfg.test_model,
            parallel_rollouts=args.parallel,
        ),
        cfg,
    )
    mean = _mean(scores)
    print(f"\nmean score: {mean:.3f} ({sum(1 for s in scores if s.passed)}/{len(scores)} passed)")
    for s in scores:
        flag = "✓" if s.passed else "✗"
        note = (s.detail.get("notes") or s.detail.get("error") or [""])
        note_str = note if isinstance(note, str) else ", ".join(note)
        print(f"  {flag} {s.row_id:30s} score={s.score:.2f}  {note_str[:80]}")


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

    eval_report_p = eval_sub.add_parser("report", help="Regenerate report from saved results")
    eval_report_p.add_argument("--results", required=True, help="Path to eval_results.json")
    eval_report_p.add_argument(
        "--output-dir", default="results", help="Output directory (default: results/)"
    )
    eval_report_p.set_defaults(func=cmd_eval_report)

    # ---- skillopt ----
    so_p = sub.add_parser(
        "skillopt", help="SkillOpt: controllable text-space skill optimizer"
    )
    so_sub = so_p.add_subparsers(dest="command", required=True)

    so_gen = so_sub.add_parser("gen-rows", help="Generate hybrid eval rows from a skill")
    so_gen.add_argument("--skill", required=True, help="Path to skill .md")
    so_gen.add_argument("--out", required=True, help="Output .jsonl path")
    so_gen.add_argument("--docs", default=None, help="Optional supplemental docs markdown file")
    so_gen.add_argument("--n", type=int, default=100, help="Number of rows to generate")
    so_gen.add_argument("--code-pct", type=int, default=55, help="Percent code rows (vs rubric)")
    _add_common_model_args(so_gen)
    so_gen.set_defaults(func=cmd_skillopt_gen_rows)

    so_run = so_sub.add_parser("run", help="Run the SkillOpt loop on a skill")
    so_run.add_argument("--skill", required=True, help="Path to skill .md")
    so_run.add_argument("--rows", required=True, help="Path to eval rows .jsonl")
    so_run.add_argument("--iters", type=int, default=15, help="Number of optimizer iterations")
    so_run.add_argument("--limit", type=int, default=0, help="Cap rows (smoke test)")
    so_run.add_argument("--parallel", type=int, default=6, help="Parallel rollouts")
    so_run.add_argument("--seed", type=int, default=42, help="Split seed")
    so_run.add_argument("--train-frac", type=float, default=0.6,
                        help="Fraction of rows for the train/failure-pool split.")
    so_run.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction for the held-out validation gate. Test = remainder.")
    so_run.add_argument(
        "--filter-baseline-below",
        type=float,
        default=0.0,
        help="Pre-pass: drop rows whose baseline score is below this (filters generator-mis-spec).",
    )
    so_run.add_argument("--lt-max", type=int, default=4,
                        help="Edit-count budget L_t at the start (paper default 4).")
    so_run.add_argument("--lt-floor", type=int, default=2,
                        help="Minimum edit-count budget L_t (paper default 2).")
    so_run.add_argument("--lt-schedule", default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="How L_t decays from lt-max to lt-floor over the run.")
    so_run.add_argument("--n-candidates", type=int, default=8,
                        help="Max candidate edits the optimizer proposes per step.")
    so_run.add_argument("--max-edit-chars", type=int, default=800,
                        help="Secondary guardrail: hard char cap on one edit's new_text.")
    _add_common_model_args(so_run)
    so_run.set_defaults(func=cmd_skillopt_run)

    so_score = so_sub.add_parser("score", help="Score a skill against rows (no optimization)")
    so_score.add_argument("--skill", required=True)
    so_score.add_argument("--rows", required=True)
    so_score.add_argument("--limit", type=int, default=0)
    so_score.add_argument("--parallel", type=int, default=6)
    _add_common_model_args(so_score)
    so_score.set_defaults(func=cmd_skillopt_score)

    return parser


def main() -> None:
    _load_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

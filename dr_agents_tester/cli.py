"""CLI entry point for datarobot-agent-tester.

Commands
--------
dr-agent agents generate   Generate/update AGENTS.md
dr-agent agents test       Evaluate AGENTS.md
dr-agent agents revise     Revise AGENTS.md from saved report

dr-agent skills test       Evaluate a skill file (or all with --all)
dr-agent skills improve    Improve a skill from its saved report
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

    return parser


def main() -> None:
    _load_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

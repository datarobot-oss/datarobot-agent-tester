<!-- AGENTS:GENERATED:START -->
# datarobot-agent-tester

Python library and CLI for generating, testing, and improving AGENTS.md files and AI coding
skills via the DataRobot LLM gateway.

## Layout

```
dr_agents_tester/   Python package (flat layout, no src/)
  __init__.py       Public API: Config, AgentsMd, Skills
  config.py         Config dataclass + env var defaults
  llm.py            litellm wrapper for DataRobot gateway
  context.py        Git file tree + priority file gathering
  agents_md.py      AgentsMd class: generate / test / revise / copilot
  skills.py         Skills class: test / improve / test_all
  cli.py            CLI entry point (dr-agent command)
skills/             Example skill files (.md) — test targets
tests/              pytest suite (no LLM calls — all mocked)
  fixtures/         Sample AGENTS.md and skill for tests
.github/workflows/
  ci.yml            Lint + test on every push/PR (3.11, 3.12, 3.13)
  agents-md.yml     Generate/test AGENTS.md and skills (needs DATAROBOT_API_TOKEN)
```

## Key files to modify

- `dr_agents_tester/agents_md.py` — AGENTS.md prompts and logic
- `dr_agents_tester/skills.py` — skill evaluation/improvement prompts and logic
- `dr_agents_tester/cli.py` — CLI subcommands
- `skills/*.md` — the actual skill files (modify or add new ones here)

Do NOT modify files in `tests/fixtures/` without updating the relevant tests.

## Commands

```bash
task install       # uv sync --all-groups
task lint          # ruff format + check + mypy (modifies files)
task lint-check    # lint check only, no modifications (used in CI)
task test          # pytest with coverage

task agents-generate   # dr-agent agents generate
task agents-test       # dr-agent agents test
task agents-revise     # dr-agent agents revise
task skills-test       # dr-agent skills test --all
task skills-improve    # dr-agent skills improve --skill <path>
```

Pass extra flags via `-- <flags>`, e.g. `task agents-generate -- --dry-run`.

## Conventions

- All LLM calls go through `llm.call_llm(prompt, model, config)` — never call litellm directly
- Tests mock `dr_agents_tester.agents_md.call_llm` and `dr_agents_tester.skills.call_llm`
- Marker constants `MARKER_START` / `MARKER_END` live in `agents_md.py` — import from there
- `Config` reads env vars by default; always call `cfg.validate()` before use in CLI commands
- Skills report files are named `<stem>.skill-report.md` (REPORT_SUFFIX constant in skills.py)
- AGENTS.md report files are named `.agents-md-report.md`

## Environment

Requires `DATAROBOT_API_TOKEN`. Copy `.env.example` to `.env` and fill in the token.
<!-- AGENTS:GENERATED:END -->

<!-- Add custom content below this line. It will be preserved when AGENTS.md is regenerated. -->

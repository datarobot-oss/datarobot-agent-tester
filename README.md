# datarobot-agent-tester

A Python library and CLI for generating, testing, and improving the two things that make AI
coding assistants actually useful:

- **AGENTS.md files** — per-directory documentation that tells agents what to modify, what
  patterns to follow, and what commands to run
- **Skill files** — markdown prompt files that define how an AI assistant should handle
  specific recurring tasks (writing commits, reviewing PRs, etc.)

Uses the [DataRobot LLM Gateway](https://docs.datarobot.com) to call LLMs for generation
and evaluation, with separate models for writing and critiquing to avoid blind spots.

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install as a library in your project
uv add datarobot-agent-tester

# Or install globally as a tool
uv tool install datarobot-agent-tester
```

To work on the library itself:

```bash
git clone https://github.com/datarobot-oss/datarobot-agent-tester
cd datarobot-agent-tester
task install
```

---

## Configuration

Set these in your `.env` file or as environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATAROBOT_API_TOKEN` | Yes | — | Your DataRobot API token |
| `DATAROBOT_ENDPOINT` | No | `https://app.datarobot.com/api/v2` | DataRobot endpoint URL |
| `AGENTS_MD_MODEL` | No | `datarobot/anthropic/claude-opus-4-5-20251101` | Model for generation/revision |
| `AGENTS_MD_TEST_MODEL` | No | `datarobot/anthropic/claude-sonnet-4-5-20250929` | Model for evaluation |
| `LLM_DEPLOYMENT_ID` | No | — | Route calls through a specific deployed model |

Copy `.env.example` to `.env` and fill in your token to get started.

---

## CLI usage

```bash
# AGENTS.md
dr-agent agents generate                   # generate/update AGENTS.md in current dir
dr-agent agents generate --dir my_service  # target a sub-directory
dr-agent agents generate --dry-run         # preview without writing
dr-agent agents test                       # evaluate AGENTS.md, save critique report
dr-agent agents revise                     # apply report feedback, rewrite AGENTS.md

# Skills
dr-agent skills test --skill skills/commit.md     # evaluate a single skill
dr-agent skills test --all                        # evaluate all skills in skills/
dr-agent skills test --all --dir my_skills/       # evaluate skills in a custom directory
dr-agent skills improve --skill skills/commit.md  # improve from saved report

# Behavioral evaluation (real coding agent in a sandbox)
dr-agent eval run-behavioral --scenarios tests/behavioral/journeys \
  --skills-pr ./skills --n-runs 3                 # drive OpenCode through scenarios
dr-agent eval sweep --prefix drat-                # dry-run sweep of leaked DR resources
dr-agent eval sweep --prefix drat- --execute      # actually delete them
```

### Behavioral evaluation

`eval run-behavioral` drives a real coding agent (OpenCode v1, pinned in
`eval/drivers/versions.py`, installed via `npm install -g opencode-ai@<version>`)
through scenario YAMLs (`kind: behavioral`) in an isolated per-run sandbox, with
skills installed the way users install them, and decides pass/fail with
programmatic `success_checks` against the DataRobot API — the LLM judge never
gates. Conditions compare `no_skill` / `skill_main` / `skill_pr` so a PR run
answers "did this change beat main, and does either beat no skill at all?".
Requires the `behavioral` extra (`pip install 'datarobot-agent-tester[behavioral]'`)
for the DataRobot checks and teardown. Verified driver behavior is documented in
[docs/opencode-driver-notes.md](docs/opencode-driver-notes.md).

### Typical AGENTS.md workflow

```bash
# 1. Generate the initial AGENTS.md
dr-agent agents generate

# 2. Review what was written, make any manual edits outside the marker block
# 3. Evaluate it — the evaluator model roleplays as a fresh agent reading the file
dr-agent agents test

# 4. Inspect .agents-md-report.md, then apply the feedback
dr-agent agents revise

# 5. Repeat steps 3-4 until verdict is GOOD
```

### Typical skills workflow

```bash
# 1. Write or copy a skill file to skills/
# 2. Evaluate it
dr-agent skills test --skill skills/my-skill.md

# 3. Review skills/my-skill.skill-report.md
# 4. Apply the feedback
dr-agent skills improve --skill skills/my-skill.md
```

---

## Python API

```python
from dr_agents_tester import Config, AgentsMd, Skills
from pathlib import Path

cfg = Config()   # reads env vars / .env automatically
cfg.validate()   # raises ValueError if DATAROBOT_API_TOKEN is missing

# AGENTS.md
agent = AgentsMd(cfg)
agent.generate(Path("."))                     # write/update AGENTS.md
report = agent.test(Path("."))                # returns and saves critique
agent.revise(Path("."))                       # apply report, rewrite file

# Sub-directory targeting
agent.generate(Path("my_service"), repo_root=Path("."))

# Skills
skills = Skills(cfg)
report = skills.test(Path("skills/commit.md"))
skills.improve(Path("skills/commit.md"))
skills.test_all(Path("skills/"))              # returns {filename: report} dict
```

---

## Included skills

The `skills/` directory contains ready-to-use skill files:

| File | Purpose |
|---|---|
| `skills/commit.md` | Write conventional git commit messages |
| `skills/review-pr.md` | Review pull requests with structured feedback |
| `skills/explain-code.md` | Explain code at the right level for the audience |

These are designed to be evaluated and improved with `dr-agent skills test --all`.

---

## How it works

### AGENTS.md generation
The generator gathers context from your repo (git-tracked file tree, README, pyproject.toml,
Taskfile, etc.) and asks an LLM to write a focused, dense AGENTS.md. Generated content is
wrapped in `<!-- AGENTS:GENERATED:START/END -->` markers so any custom content you add outside
those markers is preserved on subsequent regenerations.

After generation, optionally all AGENTS.md files in the repo are aggregated into
`.github/copilot-instructions.md` for GitHub Copilot support.

### AGENTS.md evaluation
The evaluator uses a *different* model to roleplay as a first-time agent reading the file,
then critiques it across five scenarios: orienting, making a change, running the dev loop,
avoiding mistakes, and identifying remaining gaps. This cross-model evaluation catches blind
spots that the generating model would miss when reviewing its own output.

### Skills evaluation
The evaluator assesses whether the skill prompt would produce clear, consistent results across
representative tasks, checks for ambiguity and missing edge cases, and suggests specific
improvements.

---

## Development

```bash
task install       # install dependencies
task lint          # ruff format + check + mypy
task lint-check    # lint check only (used in CI)
task test          # run pytest with coverage
```

---

## GitHub Actions

Both workflows support [`workflow_call`](https://docs.github.com/en/actions/sharing-automations/reusing-workflows),
so any repo in the org can reference them directly instead of copy-pasting.

### Using the workflows from another repo

**Minimal setup** — add two files to your repo:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request, workflow_dispatch]
jobs:
  ci:
    uses: datarobot/datarobot-agent-tester/.github/workflows/ci.yml@main
```

```yaml
# .github/workflows/agents-md.yml
name: AGENTS.md
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
jobs:
  agents-md:
    uses: datarobot/datarobot-agent-tester/.github/workflows/agents-md.yml@main
    secrets: inherit   # passes DATAROBOT_API_TOKEN automatically
```

That's it. The workflow installs `datarobot-agent-tester` from this repo automatically — no
local dependency needed.

### Available inputs

**`ci.yml`**

| Input | Default | Description |
|---|---|---|
| `python-versions` | `["3.11","3.12","3.13"]` | JSON array of Python versions to test |
| `working-directory` | `.` | Directory to run `task` commands from (for monorepos) |

```yaml
jobs:
  ci:
    uses: datarobot/datarobot-agent-tester/.github/workflows/ci.yml@main
    with:
      python-versions: '["3.11", "3.12"]'
      working-directory: my_service
```

**`agents-md.yml`**

| Input | Default | Description |
|---|---|---|
| `target-dir` | `.` | Directory to generate/test AGENTS.md for |
| `skills-dir` | `skills` | Directory containing skill files to evaluate |
| `python-version` | `3.11` | Python version to use |
| `auto-commit` | `true` | Commit updated AGENTS.md back to the repo on push to main |

```yaml
jobs:
  agents-md:
    uses: datarobot/datarobot-agent-tester/.github/workflows/agents-md.yml@main
    with:
      target-dir: my_service
      skills-dir: my_service/skills
      auto-commit: false
    secrets: inherit
```

### Required secret

Add `DATAROBOT_API_TOKEN` to your repository or organization secrets
(Settings → Secrets and variables → Actions). Using `secrets: inherit` is the simplest
way to pass it through — or pass it explicitly:

```yaml
secrets:
  DATAROBOT_API_TOKEN: ${{ secrets.DATAROBOT_API_TOKEN }}
```

### What the workflows do

**`ci.yml`** — on every push and PR:
- Static checks (`task lint-check`)
- Full test suite (`task test`)
- Python version matrix (3.11, 3.12, 3.13 by default)

**`agents-md.yml`** — requires `DATAROBOT_API_TOKEN`:
- On PRs: dry-run generate + evaluate existing AGENTS.md
- On push to main: generate updated AGENTS.md and commit it back
- All runs: evaluate every skill file in `skills-dir`

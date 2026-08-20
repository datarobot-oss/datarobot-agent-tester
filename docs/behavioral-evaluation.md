# Behavioral evaluation: user guide

`dr-agent eval run-behavioral` drives a **real coding agent** (OpenCode) through
a scenario — a realistic user prompt plus fixtures — inside an isolated
per-run sandbox, with skills installed the way users install them, then decides
pass/fail with **programmatic checks against the DataRobot API**. The LLM never
grades its own homework: checks are authoritative, trajectory metrics are
measured from the agent's own event stream.

```
scenario YAML ──► Evaluator (scenarios × conditions × k runs)
                    └─ AgentBackend
                         ├─ sandbox: fixtures + skills + isolated HOME + env allowlist
                         ├─ OpenCodeDriver: opencode run --format json  (pinned version)
                         ├─ trajectory: events → turns/tools/errors/tokens/skill triggers
                         ├─ checks: dr_project_exists, dr_deployment_healthy, …  → PASS/FAIL
                         └─ teardown: delete drat-<run_id>* resources
                    └─ stats/report: pass@k, paired t-tests, markdown + JSON
```

## Prerequisites

1. **OpenCode, pinned.** The driver refuses version drift so results stay
   attributable:

   ```bash
   npm install -g opencode-ai@1.17.11    # pin lives in eval/drivers/versions.py
   ```

2. **DataRobot credentials** in the environment or a `.env` in your working
   directory:

   ```bash
   DATAROBOT_API_TOKEN=...                                # your API token
   DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2   # default
   ```

   Runs create **real resources in this account** (use cases, datasets,
   projects, deployments), all named with a `drat-…` run prefix and deleted
   again after each run.

3. **The `behavioral` extra** (brings the DataRobot SDK for checks/teardown):

   ```bash
   uv sync --extra behavioral        # in a datarobot-agent-tester checkout
   # or: pip install 'datarobot-agent-tester[behavioral]'
   ```

## Running the golden journey

From a `datarobot-agent-skills` checkout (scenarios live there, in
`tests/behavioral/`):

```bash
task test:behavioral              # working-tree skills, k=1
task test:behavioral:baseline     # same scenario, NO skills installed
task test:behavioral:clean        # delete anything a crashed run leaked
```

Or directly, from anywhere (paths adjusted to your checkouts):

```bash
uv run --extra behavioral dr-agent eval run-behavioral \
  --scenarios ~/datarobot-agent-skills/tests/behavioral/journeys \
  --skills-pr ~/datarobot-agent-skills/skills \
  --conditions skill_pr \
  --n-runs 1 \
  --work-dir .behavioral-runs \
  --output-dir results
```

A full A/B comparison (does the PR beat main? does either beat no skill?):

```bash
uv run --extra behavioral dr-agent eval run-behavioral \
  --scenarios ~/datarobot-agent-skills/tests/behavioral/journeys \
  --skills-main ~/skills-main-checkout/skills \
  --skills-pr   ~/datarobot-agent-skills/skills \
  --n-runs 3
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--scenarios` | required | Directory of behavioral scenario YAMLs (`kind: behavioral`). Repeatable; each directory is scanned along with one subdirectory level, so `tests/behavioral/scenarios` picks up every `scenarios/<skill>/` dir. Scenario ids must be unique across all directories |
| `--skills-main` / `--skills-pr` | — | Skills dirs; each adds its condition |
| `--skip-no-skill` | off | Drop the `no_skill` baseline (included by default) |
| `--conditions` | all | Comma-separated subset to run (CI runs one per matrix job) |
| `--n-runs` | 3 | k runs per (scenario, condition) cell |
| `--difficulty` | all | Filter scenarios by difficulty |
| `--work-dir` | `.behavioral-runs` | Per-run workspaces + transcripts land here |
| `--output-dir` | `results` | `eval_report.md` + `eval_results.json` |
| `--run-prefix` | `drat-<scenario>` | Run-id prefix (CI passes `drat-gh<run>-<attempt>`) |
| `--keep-resources` | off | Skip per-run teardown — for inspecting created DR state |
| `--fail-under-pass-rate` | unset | Exit non-zero below this pass rate; unset = advisory |
| `--driver` | `opencode` | Agent driver (see `eval/drivers/`) |

### Conditions

| Condition | Question it answers |
|---|---|
| `no_skill` | Can the agent do this with just the SDK? (the "is this worth a skill?" baseline) |
| `skill_main` | Current released behavior |
| `skill_pr` | The change under review |

With ≥2 conditions the report includes paired t-tests on the 0/1 pass
indicator, matched on (scenario, run number).

### Fixture-resource injection (`requires_env` / `{env:VAR}`)

Scenarios that assert against pre-provisioned, long-lived DataRobot resources
(e.g. a fixture deployment the predictions scenario reads) declare the host
environment variables they need in `requires_env` and reference them as
`{env:VAR}` tokens in the prompt, `env` values, and string check params.
Guard rails are fail-loud at three layers: parse time (every reference must
be declared, and names must match `BEHAVIORAL_*`/`DRAT_*` — never
credentials), run start (a missing/empty variable aborts the invocation
before any agent tokens are spent), and substitution (an unresolved token
raises). The values come from the *host* environment and are substituted into
text — they are not added to the sandbox env allowlist.

## Inspecting a run

Every run gets its own directory under `--work-dir`:

```
.behavioral-runs/<run_id>/
  workspace/              # what the agent saw: fixtures, opencode.json, its outputs
  home/                   # isolated HOME (installed skills under .config/opencode/skills/)
  transcript.raw.jsonl    # the agent's full event stream, verbatim
  meta.json               # run identity, driver version/status, installed-skill digests
  trajectory.json         # normalized metrics (below)
  checks.json             # per-check PASS/FAIL with evidence strings
```

### trajectory.json

| Field | Meaning |
|---|---|
| `wall_seconds` | End-to-end agent runtime |
| `num_turns` | Model turns (assistant steps) |
| `num_tool_calls` | Tool invocations (bash/read/write/skill/…) |
| `num_errors` | Failed tool calls **plus** bash commands that exited non-zero |
| `num_retries` | Same tool re-invoked immediately after it errored — wasted work |
| `skill_triggered` / `skills_used` / `skill_first_turn` | Which skills actually fired, and when — a skill that never triggers is a `description` frontmatter bug |
| `input_tokens` / `output_tokens` / `cache_read_tokens` / `total_tokens` | From the agent's own step accounting |
| `num_unknown_events` | Events the normalizer didn't recognize (watch after an opencode version bump) |

`null` means "this driver can't report that metric", never zero.

### Reading the raw transcript

The transcript is JSONL — one event per line (`step_start`, `text`,
`tool_use`, `step_finish`). Useful recipes:

```bash
# Every tool call with status
jq -r 'select(.type=="tool_use") | "\(.part.tool)\t\(.part.state.status)"' transcript.raw.jsonl

# What the agent typed into bash
jq -r 'select(.type=="tool_use" and .part.tool=="bash") | .part.state.input.command' transcript.raw.jsonl

# Skill activations
jq -r 'select(.type=="tool_use" and .part.tool=="skill") | .part.state.input.name' transcript.raw.jsonl

# Errors only
jq -r 'select(.part.state.status=="error") | "\(.part.tool): \(.part.state.error)"' transcript.raw.jsonl

# The agent's prose
jq -r 'select(.type=="text") | .part.text' transcript.raw.jsonl
```

The full event vocabulary is documented in
[opencode-driver-notes.md](opencode-driver-notes.md).

### Reports

`--output-dir` gets `eval_report.md` (Behavioral Outcomes table: pass rate,
pass@k, efficiency means per condition; per-run PASS/FAIL with failed-check
evidence and transcript paths) and `eval_results.json` (everything, machine-
readable — this is the substrate for baselines and later optimization work).

## Cleaning up

Teardown runs automatically after every run (disable with
`--keep-resources`). For anything a crashed run leaked:

```bash
dr-agent eval sweep --prefix drat-                              # dry run (default)
dr-agent eval sweep --prefix drat- --older-than-hours 0 --execute
```

The sweeper deletes children before parents (deployments → projects →
datasets → use cases) and skips prefix-matched resources whose age it cannot
determine unless `--older-than-hours 0`.

## Writing scenarios

Scenarios live in the **skills repo** under `tests/behavioral/` — schema,
check-type table, and authoring guidelines are in its
[tests/behavioral/README.md](https://github.com/datarobot-oss/datarobot-agent-skills/tree/main/tests/behavioral).
The short version: write the prompt a real user would type, let
`env.resource_prefix: "{run_id}"` inject the naming epilogue, and put every
pass/fail assertion in `success_checks` — the `rubric` is guidance for the
(future) trajectory judge, never a gate.

## For engine developers

- **Offline tests**: `task test` — drivers are exercised through a committed
  `fake_opencode` stub and real captured event streams; no network, no agent
  binary, no SDK.
- **Live smoke** (~10 s, no DataRobot resources):
  `DRAT_LIVE=1 uv run --extra behavioral pytest tests/live/ -m live -s`
- **FakeDriver** (`eval/drivers/fake.py`) is the contract object for testing
  anything that consumes a driver.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `LAUNCH_FAILED`, "does not match pinned" | Wrong opencode version — install the pin, or pass `allow_version_drift=True` (API) for exploration |
| First run very slow / times out | Cold isolated HOME fetches npm + models.dev; usually seconds, occasionally stalls — rerun, or pre-warm |
| Tool error "user rejected permission … external_directory" | OpenCode headless auto-rejects file access outside the workspace — by design; keep fixtures and outputs inside it |
| 401 from the gateway | Token invalid/expired — check `DATAROBOT_API_TOKEN` |
| Leftover `drat-…` resources in the org | A run was killed before teardown — `dr-agent eval sweep --prefix drat- --older-than-hours 0 --execute` |
| Checks pass locally but the run "failed" | Check `meta.json` → `transcript.status`: a `timeout`/`nonzero_exit` agent can still have produced passing state; outcome is authoritative |

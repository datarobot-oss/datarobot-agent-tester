# OpenCode driver: verified behavior notes

Findings from the driver spikes (2026-08-18, opencode **1.17.11** against the
DataRobot LLM Gateway on app.datarobot.com). These back the implementation in
`dr_agents_tester/eval/drivers/` and `eval/trajectory.py`; re-verify when
bumping `eval/drivers/versions.py`.

## Gateway provider config (spike S1)

A plain `DATAROBOT_API_TOKEN` works as the API key for a custom
OpenAI-compatible provider pointed at `{DATAROBOT_ENDPOINT}/genai/llmgw`
(endpoint includes `/api/v2`). See `opencode_config.build_opencode_config` —
model ids are gateway-scoped and provider-prefixed
(`datarobot/anthropic/claude-sonnet-4-6`). Explicit `limit: {context, output}`
is included since models.dev cannot know a custom provider.

## Event stream (spike S2)

`opencode run --format json` emits one JSON object per line:

```
{"type": "<underscored>", "timestamp": ..., "sessionID": ..., "part": {"type": "<hyphenated>", ...}}
```

Observed types: `step_start`, `text` (`part.text`), `tool_use`, `step_finish`
(`part.tokens{total,input,output,reasoning,cache{read,write}}`, `part.cost` —
cost is 0 via the gateway). `tool_use` carries `part.tool` (tool name) and
`part.state`:

- `status: "completed"` → `input`, `output`, `metadata`, `time`, `title`.
  A failing bash command is still `completed`; its exit code is at
  `state.metadata.exit`.
- `status: "error"` → `input`, `error` (string), `time`. Example: reading
  outside the project dir produces
  `"The user rejected permission to use this specific tool call."`

## HOME/XDG isolation (spike S3)

`HOME` + `XDG_*` overrides are fully honored: opencode creates its config
(`~/.config/opencode`, including a `node_modules` install of the provider
package), cache (`~/.cache/opencode/models.json`), and state SQLite DB under
the overridden locations, and discovers skills from the isolated
`$XDG_CONFIG_HOME/opencode/skills/`. This is the basis of the sandbox env
allowlist in `eval/sandbox.py`.

## Skill triggering (spike S4)

Skills load through the built-in `skill` tool. Triggering is a first-class
observable: a `tool_use` event with `part.tool == "skill"` and the skill name
at `state.input.name`. The trajectory adapter derives `SKILL_TRIGGERED` events
from exactly this signature.

## First-run network cost (spike S5)

A cold isolated HOME triggers real network fetches (npm install of
`@ai-sdk/openai-compatible` into the config dir, models.dev snapshot). Usually
fast (the live smoke completes in ~7s including a cold home), but one spike
run stalled >5 minutes on the fetch — per-scenario timeouts cover this; if it
recurs in CI, pre-warm a home template once per job and copy it per run
(or build the Docker image described in the design plan).

## Headless permissions (spike S6)

Within the workspace, `write`/`read`/`bash` run unattended with default
permissions. File access *outside* the project directory triggers an
`external_directory` permission that headless mode auto-rejects (stderr:
`permission requested: external_directory (...); auto-rejecting`) — a useful
containment default we deliberately keep. Bash commands are not path-gated
the same way.

## Deferred (spike S7)

Live AutoML timings, fresh-deployment `service_health` semantics, and whether
deployment deletion requires prior deactivation get verified by the first
full golden-journey runs in datarobot-agent-skills; `dr_deployment_healthy`
already accepts `unknown` health on that basis.

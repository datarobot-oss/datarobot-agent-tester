# Skill Evaluation Report: `datarobot-agent-assist`

---

## What Works Well

### 1. Clear top-level structure
The four-category menu (Design → Code → Battle-test → Deploy) is well-defined and the activation message is exact and copy-pasteable. An agent knows precisely what to show on first contact.

### 2. Session state tracking is explicit
The state variables (`<prerequisites_passed>`, `<workspace_resolved>`, `<design_to_code>`, etc.) are enumerated with clear invalidation rules. This is unusually rigorous and reduces drift across a long conversation.

### 3. Workflow discipline rules are strong
The six numbered rules under "Workflow Discipline" — especially "No auto-advance" and "One design gate per turn" — directly address the most common failure mode of LLM agents (collapsing multi-step flows into one response).

### 4. `.env` placement rules are precise
The dedicated subsection with explicit "never in cwd" and "always pass `--target-dir`" rules prevents a common credential-leakage mistake and is actionable.

### 5. Behavioral rules section is concrete
Rules like "keep responses to 1–3 sentences during coding" and "never add performance commentary during rehearsal turns" are specific enough that two agents would behave consistently.

### 6. Error handling has a hard-stop rule
The dependency validation hard-stop with "return full output from all commands run" is unambiguous and prevents silent failures.

### 7. Timeout values are explicit
Specifying 10 min / 5 min / 30 min for different command types removes guessing.

---

## What Is Unclear or Missing

### Gap 1: Reference files are never shown — the skill is incomplete without them
The skill repeatedly says "read and follow `agent-assist-build/references/X.md`" but those files are not included. An agent following this skill cold would hit `workspace-resolution.md`, `pre-coding-checklist.md`, `dress-rehearsal.md`, etc. and have nothing to read. The skill is effectively a router to undefined content.

**Fix:** Either inline the critical reference files (especially `workspace-resolution.md`, `pre-coding-checklist.md`, `pre-deployment-checklist.md`) or add a fallback rule:
> "If a referenced file cannot be read, stop and tell the user: 'Required reference file `<path>` is missing — skill installation is incomplete.'"

---

### Gap 2: `<skill_scripts_dir>` resolution has no fallback for non-file-system agents
The instruction "resolve `<skill_scripts_dir>` as the `agent-assist-build/scripts/` subdirectory of the directory containing this `SKILL.md` file" assumes the agent can determine its own file location. In many deployment contexts (e.g., injected system prompt, API call), there is no "directory containing this file."

**Fix:** Add:
> "If the agent cannot determine the directory containing `SKILL.md` (e.g., the skill was injected as a system prompt), ask the user: 'What is the absolute path to the skill directory?' and use that as `<skill_dir>`."

---

### Gap 3: Option 3 (Battle-test) has a circular dependency
The skill says: "Read `agent-assist-simulate/SKILL.md` and jump directly to **Pre-flight Check**." But `agent-assist-simulate/SKILL.md` is also an external file not provided. If that file is missing, the agent has no fallback. Additionally, the instruction to "skip its On Activation menu" implies knowledge of that file's structure that the agent doesn't have until it reads it — creating a chicken-and-egg problem.

**Fix:** Add a guard:
> "If `agent-assist-simulate/SKILL.md` cannot be read, stop and tell the user: 'The battle-test skill is not installed. Please ensure `agent-assist-simulate/SKILL.md` exists in the skill directory.'"

---

### Gap 4: "Messy workspace" is defined elsewhere but used here without definition
The term `<design_messy_cwd>` and "messy workspaces" appear in session state and behavioral rules, but the definition is deferred to `workspace-resolution.md`. An agent reading only this file cannot determine what "messy" means (the only hint is "files other than `agent_spec.md` / `.env`" in a parenthetical).

**Fix:** Add a one-line definition inline:
> "`<design_messy_cwd>` — true when `<target_dir>` (cwd) contains files other than `agent_spec.md` and `.env` at the time design begins."

---

### Gap 5: Post-design menu choice ambiguity — what if the user types something other than 1/2/3?
The table maps `1 or "code"/"implement"`, `2 or "review"/"edit spec"`, `3 or "rehearsal"/"simulate"` but the fallback is only "re-display the menu." There's no guidance on how many re-displays before escalating, and no handling of partial matches (e.g., user says "let's build it" — is that choice 1?).

**Fix:**
> "If the user's reply does not clearly match one of the three choices after one re-display, ask: 'Did you mean (1) code, (2) review the spec, or (3) run a rehearsal?' and wait."

---

### Gap 6: Model Selection failure path is incomplete
The skill says: "In case the script fails due to any reason, do **not** proceed. Instead, return the error message to the user and ask how they want to proceed." But it doesn't define what valid responses are. Can the user skip model selection? Can they enter a model name manually?

**Fix:** Add:
> "If the user wants to proceed without model selection (e.g., they know their model ID), accept a manually entered model name and record it in the spec. If they cannot resolve the error, offer to continue with a placeholder model name (`gpt-4o` or similar) to be updated after deployment."

---

### Gap 7: Windows coding block has no alternative path
"On Windows: coding is not supported. STOP and do NOT proceed." This is a hard stop with no guidance. Users on Windows are left stranded.

**Fix:**
> "On Windows: coding is not supported in this environment. Suggest the user use GitHub Codespaces, WSL2, or a Linux/macOS machine. Provide the Codespaces link: https://github.com/features/codespaces. Then stop."

---

### Gap 8: `<dependency_check_passed>` invalidation timing is ambiguous
The skill says to invalidate when `clone_template.py`, `select_framework.py`, or `setup_template.py` runs. But it doesn't say whether invalidation happens *before* or *after* the script runs. If the script fails mid-run, is the check still invalidated?

**Fix:**
> "Invalidate `<dependency_check_passed>` (set to false) immediately **before** running any of these scripts, regardless of whether they succeed or fail."

---

### Gap 9: "After Coding" step 4 says "do not run the command yourself" — but doesn't say what to do if the user asks you to run it
A user might say "just run it for me." The skill is silent on this.

**Fix:** Add:
> "If the user asks the agent to run the local test command, decline and explain: 'The local test command starts a server process that requires an interactive terminal — it must be run by you in a new terminal window.'"

---

### Gap 10: No definition of what constitutes a "complete" spec for the Spec Display loop exit
The spec display section says "if the user indicates they are done refining (e.g. 'looks good', 'no changes', 'move on'), proceed to Agent Simulation." But there's no minimum completeness check before exiting. A user could say "looks good" on a spec with only a `name` field.

**Fix:** Add:
> "Before proceeding from Spec Display to Agent Simulation, verify the spec contains at minimum: `name`, `description`, `system_prompt`, and `llm.model`. If any are missing, note the gaps and ask the user to fill them before proceeding."

---

## Suggested Improvements

### 1. Add an inline "quick reference" state table at the top
The session state variables are scattered. A table at the top of the Session State section would help:

```markdown
| Variable | Initial Value | Set By | Cleared By |
|---|---|---|---|
| `<prerequisites_passed>` | false | Pre-requisite Check | Never (per session) |
| `<workspace_resolved>` | false | Workspace Resolution | Welcome menu reset |
| `<target_dir>` | unset | Workspace Resolution | Explicit user request |
| `<design_to_code>` | false | Post-design menu choice 1 | Welcome menu reset |
| `<design_messy_cwd>` | false | Workspace Resolution | Welcome menu reset |
| `<dependency_check_passed>` | false | Dependency validation | `<target_dir>` change or template scripts |
| `<dependency_check_target_dir>` | unset | Dependency validation | — |
```

---

### 2. Rewrite the Script Path Resolution section to handle the "no file path" case

Current text leaves agents guessing. Replace with:

```markdown
## Script Path Resolution

Before invoking any helper script, resolve `<skill_scripts_dir>` once per session:

1. Determine the directory containing this `SKILL.md` file (`<skill_dir>`).
   - If the agent has file-system access and knows the skill file path, derive `<skill_dir>` from it.
   - If the skill was injected as a system prompt or the path is unknown, ask the user:
     > "What is the absolute path to the directory containing the `SKILL.md` file?"
     Wait for the answer before continuing.
2. Set `<skill_scripts_dir>` = `<skill_dir>/agent-assist-build/scripts/`
3. Confirm it exists: run `ls <skill_scripts_dir>`. If missing, tell the user:
   > "The skill scripts directory is missing at `<skill_scripts_dir>`. Skill installation is incomplete."
   Then stop.
4. Use the resolved absolute path for every `<skill_scripts_dir>/...` reference in this skill.
```

---

### 3. Add a "Spec minimum viability" rule to Spec Display

Insert after "invite the user to refine system prompts...":

```markdown
**Minimum viable spec** — before exiting the refinement loop, the spec must contain:
- `name` — agent name
- `description` — one-sentence purpose
- `system_prompt` — at least a draft
- `llm.model` — selected from Model Selection

If any field is missing, note the gap inline and ask the user to provide it before proceeding to simulation.
```

---

### 4. Clarify the "One design gate per turn" rule with examples

The rule is stated but examples would prevent misapplication. Add:

```markdown
**Examples of violations to avoid:**
- ❌ Showing the spec draft AND asking "Ready for dress rehearsal?" in the same message
- ❌ Asking about tools AND asking about frontend in the same message
- ❌ Presenting Post-design next steps AND asking a clarifying question in the same message

**Correct pattern:** One prompt → wait for reply → next prompt.
```

---

### 5. Add explicit handling for re-entry into Design from Coding

Currently, if a user is mid-coding and says "actually I want to change the spec," the skill has no explicit path back to Design. The "Spec issues" reference in pre-coding-checklist.md is mentioned but the forward path from coding back to design is undefined.

Add to Section 2 (Coding):

```markdown
### Mid-coding spec changes
If the user requests a spec change during coding:
1. Stop coding immediately.
2. Display the current `<target_dir>/agent_spec.md`.
3. Follow [Resume Design](agent-assist-build/references/resume-design.md).
4. After spec changes are confirmed, return to [Pre-coding Checklist](#pre-coding-checklist) from step 2 (spec validation) — do not re-run workspace resolution or template clone.
```

---

## Overall Verdict

**NEEDS WORK** — The skill is architecturally sound and unusually rigorous in its state management and workflow discipline, but it is functionally incomplete: it acts as a well-structured router to a set of reference files that are not provided, leaving critical workflows (workspace resolution, pre-coding, deployment, dress rehearsal) undefined within the skill itself. Several edge cases (Windows fallback, script-path resolution in non-filesystem contexts, model selection failure, mid-coding spec changes) also lack handling. With the reference files present and the gaps above addressed, this would rate GOOD.

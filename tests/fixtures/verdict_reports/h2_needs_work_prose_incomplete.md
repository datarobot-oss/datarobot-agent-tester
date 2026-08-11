# Skill Evaluation Report: `datarobot-agent-assist`

---

## What Works Well

### 1. Clear top-level structure and entry point
The four-option menu is well-defined, and the "On Activation" section gives an agent an unambiguous starting point. The exact wording requirement for the dress rehearsal prompt is a strong practice that ensures consistency across agents.

### 2. Session state tracking is explicit and thorough
The session state section enumerates every tracked variable, their initial values, when they're set, and when they're invalidated. This is unusually rigorous and would produce consistent behavior across agents. The `.env` placement rules are especially well-specified.

### 3. Workflow Discipline rules are strong
The six numbered rules (section order, reference files, explicit skips only, no auto-advance, menus/prompts, one design gate per turn) are concrete and enforceable. These prevent the most common agent failure modes (skipping steps, combining prompts, assuming defaults).

### 4. Behavioral Rules section is actionable
Rules like "keep responses to 1–3 sentences during coding," "never add code comments unless asked," and "invoke at most one shell command per response" are specific and testable — two agents would behave consistently on these.

### 5. Error handling is well-scoped
The distinction between hard stops (dependency failures), soft retries (unexpected errors), and CRITICAL script failures is clear. The timeout table is a practical addition often missing from skills.

### 6. Clone discipline and spec validation gates are explicit
The behavioral rules section explicitly prevents common anti-patterns: treating "Code the agent" as subdirectory confirmation, running `ls` before spec validation passes, and fixing spec gaps inline instead of routing to design.

---

## What Is Unclear or Missing

### Gap 1: `agent-assist-simulate/SKILL.md` is referenced but never described
Option 3 says "Read `agent-assist-simulate/SKILL.md` and jump directly to **Pre-flight Check**" but gives no fallback if that file is missing, no description of what Pre-flight Check entails, and no path resolution guidance (unlike `<skill_scripts_dir>`). An agent encountering a missing file would have no recovery path.

**Fix:** Add a resolution rule analogous to Script Path Resolution:
> Before reading `agent-assist-simulate/SKILL.md`, confirm it exists at `<path_to_this_skill_dir>/agent-assist-simulate/SKILL.md`. If missing, tell the user the skill installation is incomplete and stop.

### Gap 2: Windows check for coding is underspecified
"On Windows: coding is not supported. STOP" — but there's no instruction on *how* to detect Windows. An agent might check `os.name`, run `uname`, or ask the user. Two agents would likely behave differently.

**Fix:**
> To detect Windows, run `python -c "import sys; print(sys.platform)"`. If the output is `win32`, stop and inform the user that coding is not supported on Windows. Direct them to use a Linux/macOS environment or GitHub Codespace.

### Gap 3: `resume-design.md` reference is circular without a clear trigger condition
The skill says "When `<target_dir>/agent_spec.md` already exists, read and follow `resume-design.md`" — but this check happens *after* workspace resolution, which itself may create or reference `<target_dir>`. It's unclear whether the existence check is before or after workspace resolution runs, and what happens if workspace resolution creates a new `<target_dir>` that happens to contain a pre-existing spec.

**Fix:** Add ordering clarity:
> After workspace resolution sets `<target_dir>`, check whether `<target_dir>/agent_spec.md` exists. If it does, read and follow `resume-design.md` instead of starting at [Clarification Phase].

### Gap 4: "At most 2 rounds of clarifying questions" is ambiguous
Does "round" mean one exchange (agent asks → user replies) or one message from the agent? If the agent asks 3 questions in one message and the user answers all, is that one round or three? Two agents would interpret this differently.

**Fix:**
> Ask **at most 2 agent messages** of clarifying questions (each may contain multiple related questions) before proposing an initial draft spec.

### Gap 5: Post-design next steps menu — unclear what happens on "3. Run dress rehearsal" if rehearsal already ran
The table says choice 3 → "Follow **[Dress Rehearsal](#dress-rehearsal)**" but doesn't specify whether this re-runs the full dress-rehearsal.md flow (including the initial prompt) or skips the intro since the user already knows what it is. An agent would guess.

**Fix:** Add a note:
> If a dress rehearsal has already run this session, skip the introductory explanation and proceed directly to the rehearsal script invocation in dress-rehearsal.md.

### Gap 6: `<design_to_code>` is set but never read
The session state defines `<design_to_code>` and it's set to `true` when the user chooses "Code the agent" — but nowhere in the skill is there a conditional that reads this variable to change behavior. Its purpose is unclear.

**Fix:** Either document what behavior it gates:
> `<design_to_code>` — used in [pre-coding-checklist.md] to skip the "do you have a spec?" prompt when transitioning directly from design.

Or remove it if it's only used in referenced files (and note that in the referenced file).

### Gap 7: No guidance on what to do if `list_llm_models.py` returns an empty list
The CRITICAL note says "if the script fails due to any reason, do not proceed" — but an empty result (success exit code, zero models) is not a failure. An agent would be stuck with no model to recommend.

**Fix:** Add:
> If the script succeeds but returns an empty model list, inform the user that no LLM models are currently available in their DataRobot environment and stop. Ask them to verify their DataRobot credentials and LLM gateway configuration before proceeding.

### Gap 8: "Messy workspace" is defined by reference but not inline
`<design_messy_cwd>` is set when "design runs in cwd with files other than `agent_spec.md` / `.env`" — but this definition is parenthetical and the actual classification logic lives in `workspace-resolution.md`. An agent reading only this file wouldn't know what "messy" means for the purpose of the clone discipline rule.

**Fix:** Add a one-line inline definition:
> A workspace is "messy" if `<target_dir>` contains files other than `agent_spec.md` and `.env` at the time of workspace resolution.

### Gap 9: Timeout rules don't specify what "timing out" means in practice
The timeout table says "return an error" — but should the agent retry, ask the user, or hard-stop? This is inconsistent with the error handling section which says "ask the user if they want to retry" for unexpected errors.

**Fix:**
> On timeout, treat it as an unexpected error: inform the user of the timeout, show the command that was running, and ask if they want to retry or stop.

---

## Suggested Improvements

### 1. Add a "quick reference" state machine diagram or table
The flow between design → rehearsal → post-design → coding is complex. A simple table would help agents stay oriented:

```
| Current state              | User says         | Next state                        |
|----------------------------|-------------------|-----------------------------------|
| Spec refinement complete   | "looks good"      | Agent Simulation (Before Coding)  |
| Dress rehearsal declined   | —                 | Post-design next steps menu       |
| Post-design menu shown     | "1" / "code"      | Pre-coding Checklist              |
| Post-design menu shown     | "2" / "edit"      | Display spec, invite changes      |
| Post-design menu shown     | "3" / "rehearsal" | Dress Rehearsal                   |
```

### 2. Clarify the "one design gate per turn" rule with examples
The current rule is abstract. Add:
> **Examples of violations:** Asking "Does the spec look good? Also, would you like a dress rehearsal?" in one message. Asking "Any changes to the tools? Ready to start coding?" in one message. Each of these combines prompts from different subsections and must be split across turns.

### 3. Make the Windows check proactive, not reactive
Currently the Windows check is at the top of section 2 — but an agent following option 1 (design) could spend a full design session before hitting this wall. Consider:
> During Pre-requisite Check, also detect the OS platform. If Windows, note that coding will not be available but design and battle-testing are still supported.

### 4. Specify what "display the spec" means when resuming
When returning from Spec issues via `resume-design.md`, the skill says "do not offer 'proceed to coding' as an alternative to refinement" — but doesn't say whether to show the full spec or just the problematic fields. Add:
> When resuming design after spec issues, display the **full** `agent_spec.md` as YAML, with a brief note indicating which fields were flagged as incomplete.

### 5. Add explicit handling for the case where the user provides a spec path, not a directory
The session state tracks `<target_dir>` as a project root, but users sometimes provide a path like `/home/user/myproject/agent_spec.md`. Add a normalization rule:
> If the user provides a file path ending in `agent_spec.md`, set `<target_dir>` to the parent directory of that file.

---

## Overall Verdict

**NEEDS WORK** — The skill is architecturally sound and unusually rigorous in its state management and workflow discipline, but several concrete gaps (missing file fallback for the simulate sub-skill, ambiguous "round" definition, unread session variable, empty model list handling, and underspecified Windows detection) would cause two agents to diverge on non-trivial edge cases that arise in realistic usage.

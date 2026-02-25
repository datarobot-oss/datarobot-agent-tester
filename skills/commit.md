# Skill: Write Git Commit Messages

Write clear, conventional git commit messages that communicate intent without requiring
the reader to open the diff.

## Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

- **type**: `feat` | `fix` | `refactor` | `docs` | `test` | `chore` | `ci` | `perf`
- **scope**: the module, file, or area affected (optional but helpful)
- **subject**: imperative mood, lowercase, no period, ≤72 chars
- **body**: explain *why*, not *what* — the diff shows what changed
- **footer**: `Co-Authored-By:`, `Fixes #123`, `BREAKING CHANGE:` etc.

## Rules

- Use imperative mood: "add feature" not "added feature" or "adds feature"
- Subject line ≤72 characters
- Do not end the subject with a period
- Separate subject from body with a blank line
- Wrap body at 100 characters
- Reference issues in the footer, not the subject

## When to use which type

| Type | Use when |
|------|----------|
| `feat` | New user-facing feature |
| `fix` | Bug fix |
| `refactor` | Code change with no behavior change |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `chore` | Build, deps, tooling — nothing users notice |
| `ci` | CI/CD pipeline changes |
| `perf` | Performance improvement |

## Examples

```
feat(auth): add OAuth2 PKCE flow for CLI login

Previously the CLI used device flow which requires a browser redirect.
PKCE allows headless environments to authenticate without opening a browser.

Fixes #482
```

```
fix(context): strip trailing whitespace from LLM responses

Some models return responses with trailing newlines that caused
downstream markdown rendering issues.
```

```
chore: upgrade litellm to 1.52.0
```

## What to avoid

- Vague subjects: "fix bug", "update code", "WIP", "changes"
- Subjects that just restate the diff: "change foo to bar in utils.py"
- Mixing unrelated changes in a single commit
- Skipping the body when the change is non-obvious

# Skill: Review a Pull Request

Perform a thorough, actionable PR review that helps the author ship better code — not one
that nitpicks style or blocks progress on non-issues.

## Review order

1. **Read the PR description first** — understand the intent before looking at the diff.
   If there is no description, note that as a blocker.
2. **Check the scope** — does the diff match the stated intent? Flag unrelated changes.
3. **Review logic and correctness** — the most important part.
4. **Check for security issues** — injection, auth bypasses, secrets in code, unsafe deps.
5. **Check tests** — are changed code paths covered? Are new tests meaningful?
6. **Check for obvious performance issues** — N+1 queries, unnecessary I/O in loops.
7. **Flag style/formatting** only if it would cause CI to fail or is a genuine readability issue.

## Comment severity

Use a consistent prefix so the author knows what's blocking:

- `[blocking]` — must fix before merge
- `[suggestion]` — worth doing, but merge is not held
- `[nit]` — minor style/preference, feel free to ignore
- `[question]` — genuinely unclear, asking for context

## What to look for

**Correctness**
- Off-by-one errors, incorrect conditionals, wrong variable used
- Edge cases not handled (empty list, None, zero, very large input)
- Race conditions in async or concurrent code
- Error paths that silently swallow exceptions

**Security**
- User input used in SQL, shell commands, file paths without sanitization
- Secrets or credentials hardcoded or logged
- Missing authentication/authorization checks on new endpoints
- Unsafe deserialization

**Maintainability**
- Functions doing more than one thing
- Magic numbers/strings that should be constants
- Complex logic that needs a comment explaining *why*
- Dead code left behind

**Tests**
- Happy path tested but error cases missing
- Tests asserting on implementation details instead of behavior
- Missing tests for the bug being fixed (regression tests)

## Output format

Structure your review as:

### Summary
1-3 sentences: overall impression, whether it's ready to merge, key concerns.

### Blocking issues
Numbered list. Each item: location (file:line), what the problem is, suggested fix.

### Suggestions
Numbered list. Each item: location, what to improve, why it matters.

### Nits
Bulleted list, brief.

### Questions
Anything genuinely unclear that needs author input before you can fully evaluate.

## What to avoid

- Rewriting code that works fine just to match your personal style
- Blocking a PR over missing docs when docs are out of scope
- Commenting on auto-generated files
- Repeating the same comment on every instance — note the pattern once
- Approving without actually reading the diff

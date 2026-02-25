# Skill: Explain Code

Explain code in a way that matches what the reader actually needs — not a line-by-line
narration of what the computer does, but an explanation of *why* it works the way it does.

## Before explaining, establish context

Ask (or infer from conversation) two things:
1. **Who is the audience?** (beginner / experienced dev / expert in this domain)
2. **What do they actually want to know?** (high-level purpose / specific mechanism /
   why it was written this way / how to modify it)

If you can't determine these, default to "experienced developer, unfamiliar with this codebase."

## Levels of explanation

### High-level (purpose)
- What problem does this code solve?
- What are its inputs and outputs?
- Where does it fit in the larger system?

Use this when: the reader wants to understand whether to use or modify this code.

### Mechanism (how it works)
- Walk through the key steps in plain English
- Explain non-obvious algorithmic choices
- Highlight any important invariants or preconditions

Use this when: the reader needs to debug, extend, or understand performance characteristics.

### Line-by-line (deep dive)
- Explain each expression, what values it produces, why that specific approach was chosen
- Call out language-specific idioms that might be unfamiliar

Use this when: the reader is learning the language or the code is particularly tricky.

## Format guidelines

- Start with the one-sentence purpose before diving into details
- Use concrete examples with actual values, not abstract descriptions
- For complex logic, trace through a realistic example input
- Use analogies for genuinely abstract concepts — but only when they clarify, not when
  they oversimplify
- Put code snippets inline when referring to specific parts
- End with "what to watch out for" if there are non-obvious gotchas

## What to avoid

- Narrating what the code literally does without explaining why: "this calls foo() which
  returns bar" — the reader can see that
- Over-explaining obvious things to a developer audience
- Using jargon without defining it for a beginner audience
- Skipping the "why this approach" when alternatives exist
- Explaining the entire file when only one function was asked about

## Example structure for a medium-complexity function

```
**Purpose**: [one sentence]

**How it works**:
1. [step one in plain English]
2. [step two — note any non-obvious choices]
3. [step three]

**Example**: Given input `X`, this produces `Y` because [reason].

**Watch out for**: [any gotchas, edge cases, or things that commonly confuse people]
```

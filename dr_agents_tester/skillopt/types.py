"""Core data types for SkillOpt."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


RowType = Literal["code", "rubric"]
EditOp = Literal["add", "delete", "replace"]


@dataclass
class EvalRow:
    """A single evaluation task for a skill.

    type="code": agent emits code; scored by sandboxed exec + call fingerprint.
    type="rubric": agent emits prose; scored by LLM judge against criteria.
    """

    id: str
    type: RowType
    prompt: str
    expected: dict[str, Any]
    source: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalRow":
        return cls(
            id=d["id"],
            type=d["type"],
            prompt=d["prompt"],
            expected=d.get("expected", {}),
            source=d.get("source", ""),
            tags=d.get("tags", []),
        )


@dataclass
class RowScore:
    """Score for a single (row, agent_output) pair. 0.0 - 1.0."""

    row_id: str
    score: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    agent_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edit:
    """A bounded add/delete/replace edit to a skill document."""

    op: EditOp
    locator: str  # heading or unique snippet identifying where to apply
    old_text: str = ""  # for delete/replace: exact substring to remove
    new_text: str = ""  # for add/replace: text to insert
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IterResult:
    """Outcome of one SkillOpt iteration (one optimization step).

    A step may apply up to L_t edits (the edit-count budget), so ``edits`` is a
    list. ``lt_budget`` is the edit-count budget in effect for this step.
    """

    iter_num: int
    edits: list[Edit]
    accepted: bool
    val_score_before: float
    val_score_after: float
    reason: str = ""  # why rejected, if rejected
    lt_budget: int = 0  # edit-count budget L_t for this step
    num_candidates: int = 0  # how many candidate edits the optimizer proposed
    num_applied: int = 0  # how many edits actually applied (others failed to locate)
    cache_hit: bool = False  # was the candidate score served from the hash-cache

    def to_dict(self) -> dict[str, Any]:
        return {
            "iter_num": self.iter_num,
            "edits": [e.to_dict() for e in self.edits],
            "accepted": self.accepted,
            "val_score_before": self.val_score_before,
            "val_score_after": self.val_score_after,
            "reason": self.reason,
            "lt_budget": self.lt_budget,
            "num_candidates": self.num_candidates,
            "num_applied": self.num_applied,
            "cache_hit": self.cache_hit,
        }

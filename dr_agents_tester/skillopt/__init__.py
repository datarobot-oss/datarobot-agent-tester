"""SkillOpt: controllable text-space optimizer for agent skills.

Implements the SkillOpt loop from Yang et al. (arxiv 2605.23904) adapted for
DataRobot skills. An optimizer model proposes bounded add/delete/replace edits
to a skill document; an edit is accepted only when it strictly improves a
held-out validation score.

Public API:
    from dr_agents_tester.skillopt import SkillOptLoop, EvalRow, Scorer
"""

from .loop import SkillOptLoop, LoopConfig
from .types import EvalRow, RowScore, IterResult, Edit
from .scorers import Scorer, MockExecScorer, RubricScorer, CompositeScorer

__all__ = [
    "SkillOptLoop",
    "LoopConfig",
    "EvalRow",
    "RowScore",
    "IterResult",
    "Edit",
    "Scorer",
    "MockExecScorer",
    "RubricScorer",
    "CompositeScorer",
]

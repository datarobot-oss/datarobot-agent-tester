"""AGENTS.md evaluation framework.

Measures whether AGENTS.md files improve AI coding agent performance
across correctness and efficiency dimensions.
"""

from .models import EvaluationReport, Scenario
from .runner import Evaluator

__all__ = ["Evaluator", "EvaluationReport", "Scenario"]

"""datarobot-agent-tester — Generate, test, and improve AGENTS.md files and AI coding skills.

Quick start::

    from dr_agents_tester import Config, AgentsMd, Skills
    from pathlib import Path

    cfg = Config()   # reads DATAROBOT_API_TOKEN, DATAROBOT_ENDPOINT from env
    cfg.validate()

    # AGENTS.md workflow
    agent = AgentsMd(cfg)
    agent.generate(Path("."))           # write/update AGENTS.md
    agent.test(Path("."))               # evaluate → saves .agents-md-report.md
    agent.revise(Path("."))             # apply report feedback

    # Skills workflow
    skills = Skills(cfg)
    skills.test(Path("skills/commit.md"))         # evaluate → saves commit.skill-report.md
    skills.improve(Path("skills/commit.md"))      # apply report feedback
    skills.test_all(Path("skills/"))              # test every .md in the directory
"""

from .agents_md import AgentsMd
from .config import Config
from .eval import Evaluator
from .skills import Skills

# pytest_plugin is intentionally NOT imported here — it requires pytest which
# is a dev-only dependency.  Import directly when needed:
#   from dr_agents_tester.pytest_plugin import make_skill_e2e_test

__version__ = "0.1.0"
__all__ = ["AgentsMd", "Config", "Evaluator", "Skills", "__version__"]

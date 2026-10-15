"""Pinned agent-CLI versions — the single source of truth.

The driver refuses to run against a different version (unless explicitly
overridden) and CI installs exactly this version, so behavioral results are
always attributable to a known agent build.
"""

OPENCODE_PINNED_VERSION = "1.17.11"

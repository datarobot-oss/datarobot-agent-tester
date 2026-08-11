"""Root conftest: enables the `pytester` fixture for testing the pytest plugin.

`pytest_plugins` is only honored in the rootdir conftest, which is why this
lives here rather than in tests/conftest.py.
"""

pytest_plugins = ["pytester"]

# Skill: Write Unit Tests

Write focused, readable unit tests that verify behavior without over-specifying implementation.

## Rules
- Test one behavior per test function
- Name tests as `test_<what>_<when>_<expected>`
- Use `pytest` and `pytest-mock` for mocking
- Prefer `assert` statements over unittest-style methods
- Mock at the boundary — mock external I/O, not internal functions

## Structure
```python
def test_<subject>_<scenario>():
    # Arrange
    ...
    # Act
    result = ...
    # Assert
    assert result == expected
```

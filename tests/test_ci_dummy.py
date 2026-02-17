"""Dummy test that intentionally fails to confirm CI pipeline fails on test failure.
Remove this file after verifying CI behavior."""


def test_ci_fails_on_failure():
    """Intentionally fail to verify CI catches failing tests."""
    assert False, "Dummy failure: remove tests/test_ci_dummy.py after confirming CI"

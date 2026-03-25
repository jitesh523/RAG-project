import os
import importlib
import contextlib
import types
import pytest


@contextlib.contextmanager
def _env(**env):
    old = {k: os.environ.get(k) for k in env.keys()}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def reload_app_with_env(**env) -> types.ModuleType:
    """Set env vars, then reload config and app modules so settings take effect."""
    with _env(**env):
        # Reload config first so class attributes rebind from env
        import src.config as cfg

        importlib.reload(cfg)
        # Now reload app module to rebuild routes/middleware
        import src.app.fastapi_app as appmod

        importlib.reload(appmod)
        return appmod


@pytest.fixture
def appmod_factory():
    return reload_app_with_env


@pytest.fixture(autouse=True)
def default_metrics_public_true(monkeypatch):
    """Default tests to METRICS_PUBLIC=true unless a test overrides via reload_app_with_env."""
    monkeypatch.setenv("METRICS_PUBLIC", "true")
    yield

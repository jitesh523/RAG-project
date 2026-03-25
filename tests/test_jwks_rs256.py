from fastapi.testclient import TestClient
from tests.conftest import reload_app_with_env


def test_metrics_allows_valid_rs256_jwt(monkeypatch):
    # Fake JWKS with a single RSA key (public only); use kid=abc
    # For simplicity in unit test, we will mock requests.get to return this JWKS
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "abc",
                "e": "AQAB",
                "n": "sX3K1wI4Kk9jrafakefakefakefakefakefakefakefakefakefakefakefakefakefakefakefake",
            }
        ]
    }

    class FakeResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, timeout):
        return FakeResp(jwks)

    # Build an RS256 token header with kid=abc and dummy body; we won't validate signature fully here
    # We only test that our code path attempts JWKS and accepts when decode doesn't raise.
    # To avoid real signature validation, we will monkeypatch jwt.decode to just return claims if RS256 path is taken.
    def fake_decode(token, key=None, algorithms=None, issuer=None, audience=None):
        return {"iss": issuer, "aud": audience}

    import src.app.fastapi_app as appmod

    monkeypatch.setattr(appmod.requests, "get", fake_get)
    monkeypatch.setattr(appmod.jwt, "decode", fake_decode)

    # Header with kid=abc, payload arbitrary, signature ignored by fake_decode
    token = "header.payload.signature"

    appmod = reload_app_with_env(
        JWT_ALG="RS256", JWT_JWKS_URL="https://example.com/.well-known/jwks.json"
    )
    client = TestClient(appmod.app)

    r = client.get("/metrics", headers={"authorization": f"Bearer {token}"})
    assert r.status_code == 200

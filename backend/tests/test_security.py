"""Unit tests for password hashing and JWT handling."""

from __future__ import annotations

import time

import pytest

from app.core.exceptions import AuthenticationError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_is_salted_and_verifies():
    first = hash_password("CorrectHorse1")
    second = hash_password("CorrectHorse1")
    assert first != second
    assert verify_password("CorrectHorse1", first)
    assert verify_password("CorrectHorse1", second)


def test_verify_rejects_wrong_password():
    assert not verify_password("wrong", hash_password("CorrectHorse1"))


def test_verify_handles_malformed_hash():
    assert not verify_password("anything", "not-a-bcrypt-hash")


def test_access_token_round_trip():
    token = create_access_token("user-1", {"email": "a@b.test"})
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == "user-1"
    assert payload["email"] == "a@b.test"


def test_token_type_is_enforced():
    refresh = create_refresh_token("user-1")
    with pytest.raises(AuthenticationError):
        decode_token(refresh, expected_type="access")


def test_tampered_token_is_rejected():
    token = create_access_token("user-1")
    tampered = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
    with pytest.raises(AuthenticationError):
        decode_token(tampered)


def test_tokens_have_unique_jti():
    first = decode_token(create_access_token("user-1"))
    time.sleep(0.01)
    second = decode_token(create_access_token("user-1"))
    assert first["jti"] != second["jti"]

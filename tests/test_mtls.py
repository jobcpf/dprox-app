from __future__ import annotations

import pytest
from cryptography.x509.oid import ExtendedKeyUsageOID

from dprox.mtls import AuthFailure, validate_client_cert
from tests.cert_helpers import (
    make_client_cert_der,
    make_multi_cn_der,
    make_no_cn_der,
)


def test_valid_agent_cert_returns_cn_and_serial() -> None:
    der = make_client_cert_der(cn="agent_alice")
    cn, serial = validate_client_cert(der)
    assert cn == "agent_alice"
    assert serial  # hex string, non-empty
    int(serial, 16)  # parses as hex


def test_no_cert_raises_auth_required() -> None:
    with pytest.raises(AuthFailure) as exc:
        validate_client_cert(None)
    assert exc.value.code == "auth_required"
    assert exc.value.status == 401


def test_garbage_bytes_raise_cert_invalid() -> None:
    with pytest.raises(AuthFailure) as exc:
        validate_client_cert(b"not a real cert")
    assert exc.value.code == "cert_invalid"


def test_cert_without_eku_extension_raises_cert_invalid() -> None:
    der = make_client_cert_der(cn="agent_alice", omit_eku=True)
    with pytest.raises(AuthFailure) as exc:
        validate_client_cert(der)
    assert exc.value.code == "cert_invalid"
    assert "no extKeyUsage" in exc.value.message


def test_cert_with_only_server_auth_eku_raises_cert_invalid() -> None:
    der = make_client_cert_der(
        cn="dprox-arc", eku=[ExtendedKeyUsageOID.SERVER_AUTH]
    )
    with pytest.raises(AuthFailure) as exc:
        validate_client_cert(der)
    assert exc.value.code == "cert_invalid"
    assert "clientAuth" in exc.value.message


def test_cert_with_client_auth_among_other_eku_accepted() -> None:
    """A cert with both clientAuth AND other usages is fine — clientAuth is what we require."""
    der = make_client_cert_der(
        cn="agent_alice",
        eku=[ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH],
    )
    cn, _ = validate_client_cert(der)
    assert cn == "agent_alice"


def test_cert_with_two_cn_attributes_raises_cn_unparseable() -> None:
    der = make_multi_cn_der(cns=["agent_alice", "admin_alice"])
    with pytest.raises(AuthFailure) as exc:
        validate_client_cert(der)
    assert exc.value.code == "cn_unparseable"
    assert "2 CN" in exc.value.message


def test_cert_with_no_cn_attribute_raises_cn_unparseable() -> None:
    der = make_no_cn_der()
    with pytest.raises(AuthFailure) as exc:
        validate_client_cert(der)
    assert exc.value.code == "cn_unparseable"


def test_cn_value_is_returned_byte_exact() -> None:
    """Must match agent_users[].name byte-for-byte (cn_equals_name strategy)."""
    der = make_client_cert_der(cn="agent_oversight")
    cn, _ = validate_client_cert(der)
    assert cn == "agent_oversight"
    assert len(cn) == len("agent_oversight")


def test_serial_is_hex_string() -> None:
    der = make_client_cert_der(cn="agent_alice")
    _, serial = validate_client_cert(der)
    assert all(c in "0123456789abcdef" for c in serial.lower())


def test_auth_failure_carries_status_code() -> None:
    """AuthFailure subclasses can override status; default 401."""
    exc = AuthFailure("test_code", "test message")
    assert exc.status == 401
    custom = AuthFailure("test_code", "test message", status=403)
    assert custom.status == 403

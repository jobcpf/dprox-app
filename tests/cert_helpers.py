"""In-process cert generation for unit tests.

Returns DER bytes (what ``DproxHttpProtocol`` injects into the ASGI scope)
so tests can call ``validate_client_cert`` directly without touching disk
or running TLS handshakes.
"""

from __future__ import annotations

import datetime as dt

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _gen_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def make_ca() -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    key = _gen_key()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert, key


def make_client_cert_der(
    *,
    cn: str = "agent_alice",
    ca: tuple[x509.Certificate, rsa.RSAPrivateKey] | None = None,
    eku: list[x509.ObjectIdentifier] | None = None,
    extra_subject_attrs: list[x509.NameAttribute] | None = None,
    omit_eku: bool = False,
) -> bytes:
    """Build a leaf cert and return its DER bytes.

    Defaults produce a valid agent client cert (single CN, EKU=clientAuth)
    that ``validate_client_cert`` should accept. Override the kwargs to
    construct certs that exercise specific failure paths.
    """
    if ca is None:
        ca = make_ca()
    ca_cert, ca_key = ca

    subject_attrs: list[x509.NameAttribute] = [
        x509.NameAttribute(NameOID.COMMON_NAME, cn)
    ]
    if extra_subject_attrs:
        subject_attrs.extend(extra_subject_attrs)

    leaf_key = _gen_key()
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name(subject_attrs))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    )

    if not omit_eku:
        usages = eku or [ExtendedKeyUsageOID.CLIENT_AUTH]
        builder = builder.add_extension(x509.ExtendedKeyUsage(usages), critical=False)

    cert = builder.sign(ca_key, hashes.SHA256())
    return cert.public_bytes(serialization.Encoding.DER)


def make_multi_cn_der(*, cns: list[str]) -> bytes:
    """Build a cert with multiple CN attributes (rejected by validate_client_cert)."""
    ca = make_ca()
    ca_cert, ca_key = ca
    leaf_key = _gen_key()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name) for name in cns])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)


def make_no_cn_der() -> bytes:
    """Build a cert with no CN attribute (only an O attribute)."""
    ca = make_ca()
    ca_cert, ca_key = ca
    leaf_key = _gen_key()
    subject = x509.Name(
        [x509.NameAttribute(NameOID.ORGANIZATION_NAME, "no-cn-here")]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)

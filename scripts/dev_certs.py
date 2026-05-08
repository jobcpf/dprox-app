#!/usr/bin/env python3
"""Generate a throwaway CA + server + agent certs for local mTLS dev.

Cross-platform; uses ``cryptography`` (already a dprox dependency) so no
openssl required. Writes PEM files to ``./certs/`` relative to the current
working directory — match this in ``examples/config.dev.yml``.

Usage (from the dprox-app/ directory):

    python scripts/dev_certs.py
    dprox serve --config examples/config.dev.yml
    curl --cert ./certs/agent_alice.crt --key ./certs/agent_alice.key \\
         --cacert ./certs/ca.crt \\
         https://localhost:8443/v1/query \\
         -H 'Content-Type: application/json' \\
         -d '{"query": "test", "limit": 5}'

NOT for production. The platform's Ansible owns real cert provisioning
on zaphod (see ``cert-provisioning-brief.md``).
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

OUT_DIR = Path("certs")
CA_VALIDITY_DAYS = 3650
LEAF_VALIDITY_DAYS = 90

# Each agent listed here gets a client cert. Match
# `examples/compiled_plan.example.yml`.
AGENTS = ["agent_alice", "agent_bob", "agent_oversight"]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _gen_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _write_key(path: Path, key: rsa.RSAPrivateKey) -> None:
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)


def _write_cert(path: Path, cert: x509.Certificate) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _make_ca(out: Path) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    print("[*] generating CA (CN=dprox-dev-ca)")
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "dprox-dev-ca")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=CA_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    _write_key(out / "ca.key", key)
    _write_cert(out / "ca.crt", cert)
    return cert, key


def _make_server(
    out: Path, ca_cert: x509.Certificate, ca_key: rsa.RSAPrivateKey
) -> None:
    print("[*] generating server cert (CN=dprox-dev, EKU=serverAuth)")
    key = _gen_key()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "dprox-dev")]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=LEAF_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.DNSName("dprox-dev"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(out / "server.key", key)
    _write_cert(out / "server.crt", cert)


def _make_agent(
    name: str,
    out: Path,
    ca_cert: x509.Certificate,
    ca_key: rsa.RSAPrivateKey,
) -> None:
    print(f"[*] generating client cert (CN={name}, EKU=clientAuth)")
    key = _gen_key()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + dt.timedelta(days=LEAF_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(out / f"{name}.key", key)
    _write_cert(out / f"{name}.crt", cert)


def main() -> int:
    out = OUT_DIR
    out.mkdir(exist_ok=True)
    ca_cert, ca_key = _make_ca(out)
    _make_server(out, ca_cert, ca_key)
    for agent in AGENTS:
        _make_agent(agent, out, ca_cert, ca_key)
    print(f"[*] done — {len(list(out.glob('*.crt')))} certs in {out.resolve()}")
    print("    Now run: dprox serve --config examples/config.dev.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

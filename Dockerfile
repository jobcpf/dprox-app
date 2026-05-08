# syntax=docker/dockerfile:1.7
#
# dprox v0.1 production image. Single-stage; cryptography ships
# manylinux wheels for python:3.12-slim so no build toolchain needed.
#
# Per dprox-design-spec-v0.2.md §11/§12:
#   - python:3.12-slim base
#   - ca-certificates installed (mTLS chain validation, no other system deps)
#   - drops to dprox user (UID/GID 10042) before CMD
#   - dprox serve as CMD; reads DPROX_CONFIG (default /etc/dprox/config.yml)
#
# Certs are bind-mounted at /etc/dprox/certs/ by the platform's Ansible.
# The platform must chown those mount points to UID 10042 so dprox can read
# server.key (mode 0400, owner-only). See README.md §"Cert mount permissions".

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10042 dprox \
    && useradd --system --uid 10042 --gid dprox --no-create-home --shell /usr/sbin/nologin dprox

WORKDIR /app

# Copy install metadata first so dependency resolution caches between rebuilds.
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --no-compile . \
    && python -c "import dprox; print('dprox', dprox.__version__)"

ENV DPROX_CONFIG=/etc/dprox/config.yml \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER dprox

EXPOSE 8443

CMD ["dprox", "serve"]

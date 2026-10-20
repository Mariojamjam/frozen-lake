FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Pinned uv from the official image (keep in sync with .github/workflows/ci.yml).
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /usr/local/bin/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    COLORTERM=truecolor

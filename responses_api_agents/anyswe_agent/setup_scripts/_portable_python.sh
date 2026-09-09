#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"
PBS_RELEASE="${PBS_RELEASE:-20260805}"
# musl (not gnu): statically linked, so it has no dependency on the target task image's glibc.
# SWE-bench Pro task images span many distros/ages; the gnu build was relocating against
# posix_fallocate64 missing in some of them (return_code=127) and failing to even launch in
# others, dropping those instances to 0 reward before a single model call.
ARCH="${ARCH:-x86_64-unknown-linux-musl}"

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        echo "Portable python already present at $DEPS_DIR/bin/python3"
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${ARCH}-install_only.tar.gz"
    echo "Downloading portable python: $url"
    # Tarball extracts to python/{bin,lib}.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    if portable_python_can_run; then
        install_python_packages --upgrade pip
    fi
}

portable_python_can_run() {
    "$DEPS_DIR/bin/python3" -c "" >/dev/null 2>&1
}

install_python_packages() {
    if portable_python_can_run; then
        "$DEPS_DIR/bin/python3" -m pip install "$@"
        return
    fi
    command -v uv >/dev/null || { echo "uv is required to prepare a cross-platform runtime" >&2; return 1; }
    uv pip install \
        --prefix "$DEPS_DIR" \
        --python-version "$PYTHON_VERSION" \
        --python-platform "$ARCH" \
        "$@"
}

install_nemo_gym_deps() {
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    install_python_packages "$NEMO_GYM_ROOT"
}

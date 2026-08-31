#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

PYTHON_VERSION="${PYTHON_VERSION:-3.12.8}"
PBS_RELEASE="${PBS_RELEASE:-20241219}"
ARCH="${ARCH:-x86_64-unknown-linux-gnu}"

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        echo "Portable python already present at $DEPS_DIR/bin/python3"
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${ARCH}-install_only.tar.gz"
    echo "Downloading portable python: $url"
    # Tarball extracts to python/{bin,lib}.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    "$DEPS_DIR/bin/python3" -m pip install --upgrade pip
}

install_nemo_gym_deps() {
    # Install NeMo-Gym runtime deps; live source is mounted separately.
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    "$DEPS_DIR/bin/python3" -m pip install "$NEMO_GYM_ROOT"
}

# $DEPS_DIR is later bind-mounted read-only into every task sandbox at this fixed path
# (see app.py's `f"{params.agent_deps_dir}:/agent_deps_mount:ro"`), and PATH is set to put
# /agent_deps_mount/bin first -- so the model's own terminal/execute_code calls can resolve
# `pip` (etc.) here too. But every console-script pip installs bakes in $DEPS_DIR's absolute
# *build-time* path, not the mount path it's later relocated to -- as a `#!` line when short
# enough, otherwise (pip itself included) as a `'''exec' <path> "$0" "$@"` sh-bootstrap wrapper
# on the second line, since shebang lines have an OS length limit these long paths can exceed.
# If that build-time path isn't reachable from inside the sandbox, the script fails with a
# plain "not found" even though nothing is actually missing. Rewrite every literal occurrence
# of $DEPS_DIR in $DEPS_DIR/bin to the portable mount path so scripts survive the relocation.
fixup_relocated_shebangs() {
    local target="/agent_deps_mount"
    local f
    for f in "$DEPS_DIR"/bin/*; do
        [ -f "$f" ] || continue
        grep -qF "$DEPS_DIR" "$f" 2>/dev/null || continue
        sed -i "s|$DEPS_DIR|$target|g" "$f"
    done
}

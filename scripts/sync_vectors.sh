#!/usr/bin/env bash
# Vendor sdk-conformance's shared unit-test vectors into tests/vectors/.
#
#   ./scripts/sync_vectors.sh ../sdk-conformance
#
# Check the checkout out at the commit CI pins (the `ref` of the conformance job in
# .github/workflows/ci.yaml) first: CI fails unless tests/vectors/ is byte-identical to
# that commit's vectors/. A checkout with uncommitted changes under vectors/ is refused,
# since no commit holds what would be copied.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <sdk-conformance checkout>" >&2
    exit 2
fi

SOURCE="$1/vectors"
TARGET="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/vectors"

if [ ! -f "$SOURCE/README.md" ]; then
    echo "error: $SOURCE is not sdk-conformance's vectors/ directory" >&2
    exit 1
fi

if git -C "$1" rev-parse --git-dir >/dev/null 2>&1; then
    if [ -n "$(git -C "$1" status --porcelain -- vectors)" ]; then
        echo "error: $SOURCE has uncommitted changes" >&2
        exit 1
    fi
    echo "Vendoring vectors/ from sdk-conformance@$(git -C "$1" rev-parse HEAD)"
fi

rm -rf "$TARGET"
cp -R "$SOURCE" "$TARGET"
diff -r "$TARGET" "$SOURCE"
echo "Copied to $TARGET"

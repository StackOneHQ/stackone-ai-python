#!/usr/bin/env bash
# Validate the SDK against everything that consumes it.
#
#   ./scripts/validate.sh                 # all sections
#   ./scripts/validate.sh examples        # one of: examples | conformance | sdk | pydantic | adk
#
# Sibling repo locations can be overridden:
#   CONFORMANCE=/path/to/sdk-conformance ADK=/path/to/stackone-adk-plugin ./scripts/validate.sh
#
# Everything here runs against the sdk-conformance mock API on 127.0.0.1 with a
# dummy key. Nothing touches the live StackOne API, so results are deterministic
# and no credentials are needed.
#
# A section that cannot run reports SKIP and says why. Skips are never counted as
# passes: the exit code is 0 only if every section that ran passed, and the summary
# always prints what was skipped.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="$(dirname "$HERE")"
CONFORMANCE="${CONFORMANCE:-$SDK/../sdk-conformance}"
ADK="${ADK:-$SDK/../adk-26-ci}"

failed=0
declare -a PASSED=() FAILED=() SKIPPED=()


section() { printf '\n\033[1m################ %s ################\033[0m\n' "$1"; }
pass()    { PASSED+=("$1");  printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail()    { FAILED+=("$1");  failed=1; printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
skip()    { SKIPPED+=("$1 — $2"); printf '  \033[33mSKIP\033[0m  %s — %s\n' "$1" "$2"; }

# --- examples -----------------------------------------------------------------
# Static checks only. Examples guard their body behind `if __name__ == "__main__":`,
# so importing one runs its module-level code and proves the imports resolve; type
# checking covers the body. Neither needs credentials, and neither calls StackOne.
validate_examples() {
    section "examples"

    local ok=1
    for f in "$SDK"/examples/*.py; do
        if (cd "$SDK" && uv run --quiet python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_probe', '$f')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
" >/dev/null 2>&1); then
            :
        else
            fail "example imports: $(basename "$f")"
            ok=0
        fi
    done
    [ "$ok" = 1 ] && pass "all examples import cleanly"

    # Catches use of a removed or renamed SDK symbol inside a function body,
    # which importing alone cannot see.
    #
    # Score ty's exit code. This used to grep for a diagnostic line mentioning
    # `stackone_ai`, but ty does not name the package in the message it emits for
    # the case that matters — `Object of type `Tools` has no attribute `gone``
    # contains no "stackone_ai" at all — so the check passed vacuously on exactly
    # the breakage it was written to catch, and the trailing `|| true` discarded
    # every other type error in examples/ besides.
    if (cd "$SDK" && uv run ty check examples/); then
        pass "examples type-check against the current SDK"
    else
        fail "examples type-check against the current SDK"
    fi

}

# --- conformance --------------------------------------------------------------
validate_conformance() {
    section "conformance (wire contract)"

    if [ ! -d "$CONFORMANCE" ]; then
        skip "conformance" "repo not found at $CONFORMANCE (set CONFORMANCE=...)"
        return
    fi
    if [ ! -x "$CONFORMANCE/node_modules/.bin/tsx" ]; then
        skip "conformance" "tsx missing — run 'pnpm install' in $CONFORMANCE"
        return
    fi

    # --strict-schema is the acceptance gate: it fails on any schema keyword the
    # SDK drops between what the server serves and what the model is shown.
    if (cd "$CONFORMANCE" && PYTHON_SDK_DIR="$SDK" pnpm --silent test:python -- --strict-schema); then
        pass "conformance wire contract + strict schema pass-through"
    else
        fail "conformance wire contract + strict schema pass-through"
    fi
}

# --- consumer smoke tests -----------------------------------------------------
smoke() {
    local target="$1" label="$2"

    if [ ! -d "$CONFORMANCE" ]; then
        skip "$label" "conformance repo not found at $CONFORMANCE"
        return
    fi
    if [ "$target" = "adk" ] && [ ! -d "$ADK" ]; then
        skip "$label" "ADK plugin not found at $ADK (set ADK=...)"
        return
    fi

    local out
    out="$(cd "$CONFORMANCE" && SDK_PY="$SDK" ADK="$ADK" ./scripts/run_smoke.sh "$target" 2>&1)"
    if [ $? -eq 0 ]; then
        printf '%s\n' "$out"
        pass "$label"
        return
    fi
    printf '%s\n' "$out"

    # A consumer pinned to an unpublished major cannot be installed, which says
    # nothing about this SDK. Report it as not-run rather than as a failure here;
    # it resolves itself once that version is on PyPI.
    if grep -q 'only the following versions of stackone-ai are available' <<<"$out"; then
        skip "$label" "consumer pins a stackone-ai version that is not on PyPI yet"
        return
    fi
    fail "$label"
}

want="${1:-all}"
case "$want" in
    all | examples | conformance | sdk | pydantic | adk) ;;
    *)
        echo "unknown target '$want' — expected one of: all, examples, conformance, sdk, pydantic, adk" >&2
        exit 2
        ;;
esac

[ "$want" = all ] || [ "$want" = examples ]    && validate_examples
[ "$want" = all ] || [ "$want" = conformance ] && validate_conformance
[ "$want" = all ] || [ "$want" = sdk ]         && smoke sdk      "smoke: SDK"
[ "$want" = all ] || [ "$want" = pydantic ]    && smoke pydantic "smoke: Pydantic AI (1.x and 2.x)"
[ "$want" = all ] || [ "$want" = adk ]         && smoke adk      "smoke: Google ADK"

section "summary"
printf '  passed:  %d\n' "${#PASSED[@]}"
printf '  failed:  %d\n' "${#FAILED[@]}"
printf '  skipped: %d\n' "${#SKIPPED[@]}"
for s in "${SKIPPED[@]:-}"; do [ -n "$s" ] && printf '    \033[33m- %s\033[0m\n' "$s"; done
for f in "${FAILED[@]:-}"; do [ -n "$f" ] && printf '    \033[31m- %s\033[0m\n' "$f"; done

echo
if [ "$failed" -ne 0 ]; then
    echo -e "\033[31mVALIDATE: FAIL\033[0m"
    exit 1
fi
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo -e "\033[32mVALIDATE: PASS\033[0m (with ${#SKIPPED[@]} skipped — see above)"
    exit 0
fi
echo -e "\033[32mVALIDATE: PASS\033[0m"

#!/usr/bin/env bash
# Validate the SDK against everything that consumes it.
#
#   ./scripts/validate.sh                 # all sections
#   ./scripts/validate.sh examples        # one of: examples | conformance | sdk | pydantic | adk
#
# Sibling repo locations can be overridden:
#   CONFORMANCE=/path/to/sdk-conformance ADK=/path/to/stackone-adk-plugin ./scripts/validate.sh
#
# A section that cannot run reports SKIP and says why. Skips are never counted as
# passes: the exit code is 0 only if every section that ran passed, and the summary
# always prints what was skipped.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="$(dirname "$HERE")"
CONFORMANCE="${CONFORMANCE:-$SDK/../sdk-conformance}"
ADK="${ADK:-$SDK/../stackone-adk-plugin}"

failed=0
declare -a PASSED=() FAILED=() SKIPPED=()

section() { printf '\n\033[1m################ %s ################\033[0m\n' "$1"; }
pass()    { PASSED+=("$1");  printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail()    { FAILED+=("$1");  failed=1; printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
skip()    { SKIPPED+=("$1 — $2"); printf '  \033[33mSKIP\033[0m  %s — %s\n' "$1" "$2"; }

# --- examples -----------------------------------------------------------------
# Examples guard their body behind `if __name__ == "__main__":`, so importing one
# runs only its module-level code. Importing therefore proves the imports resolve;
# it does NOT prove the body works. Type checking covers the body without needing
# credentials, and the live run is attempted only when credentials are present.
validate_examples() {
    section "examples"

    local ok=1
    for f in "$SDK"/examples/*.py; do
        [ "$(basename "$f")" = "test_examples.py" ] && continue
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
    # Capture first, then grep. Piping ty straight into grep would be scored by
    # `set -o pipefail`, which returns ty's non-zero exit even when grep matched —
    # so a successful detection reads as "no match" and the check passes vacuously.
    # Match the diagnostic line itself, not any line mentioning the package: ty
    # echoes source context, so a bare `stackone_ai` grep matches every example's
    # own import and reports a failure on a perfectly clean tree.
    local ty_out ty_hits
    ty_out="$(cd "$SDK" && uv run ty check examples/ 2>&1)" || true
    ty_hits="$(grep -E '^error\[unresolved-(import|attribute)\]:.*stackone_ai' <<<"$ty_out" || true)"
    if [ -n "$ty_hits" ]; then
        fail "examples reference a stackone_ai symbol that does not exist"
        sed 's/^/        /' <<<"$ty_hits"
    else
        pass "no example references a missing stackone_ai symbol"
    fi

    if [ -n "${STACKONE_API_KEY:-}" ] && [ -n "${STACKONE_ACCOUNT_ID:-}" ]; then
        local ran=1
        for f in "$SDK"/examples/*.py; do
            [ "$(basename "$f")" = "test_examples.py" ] && continue
            if ! (cd "$SDK" && uv run --quiet python "$f" >/dev/null 2>&1); then
                fail "example live run: $(basename "$f")"
                ran=0
            fi
        done
        [ "$ran" = 1 ] && pass "all examples run against the live API"
    else
        skip "examples live run" "STACKONE_API_KEY / STACKONE_ACCOUNT_ID not set"
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

    if (cd "$CONFORMANCE" && SDK_PY="$SDK" ADK="$ADK" ./scripts/run_smoke.sh "$target"); then
        pass "$label"
    else
        fail "$label"
    fi
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

#!/usr/bin/env bash
# Proof-of-gate (make red): the regression gate MUST fail on each sabotage, and name the
# failing metric in stderr. Any case that does not fail, or does not name its metric, fails
# this script. A gate you cannot see fail is not a gate.
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
out="$tmp/r.json"
empty="$tmp/empty.jsonl"
: >"$empty"
tiny="$tmp/tiny.jsonl"
printf '%s\n' '{"id":"t1","question":"q","answer":"1","required_tools":["retrieve"],"gold_facts":["x"]}' >"$tiny"

status=0

# $1 human label, $2 expected exit code, $3 metric token expected in stderr, rest: eval args
check() {
  local label="$1" want_code="$2" want_metric="$3"
  shift 3
  local err
  err="$(uv run ailab-eval --output "$out" "$@" 2>&1 >/dev/null)"
  local code=$?
  if [[ "$code" -ne "$want_code" ]]; then
    echo "FAIL [$label]: exit $code, expected $want_code" >&2
    echo "$err" | sed 's/^/    /' >&2
    status=1
  elif ! grep -qiE "$want_metric" <<<"$err"; then
    echo "FAIL [$label]: stderr did not name '$want_metric'" >&2
    echo "$err" | sed 's/^/    /' >&2
    status=1
  else
    echo "PASS [$label]: exit $code, named '$want_metric'"
    grep -iE "REGRESSION|error" <<<"$err" | sed 's/^/    /'
  fi
}

echo "== make red: proving the gate fails on each sabotage =="
check "1 stub below task_success floor" 1 "task_success"         --min-task-success 0.6
check "2 no tools (answers blind)"      1 "tool_usage"            --no-tools --min-task-success 0.6
check "3 hardcoded answer"              1 "non_triviality"        --hardcode-answer --min-task-success 0.6
check "4 forced invalid actions"        1 "invalid_action_rate"   --force-invalid --min-task-success 0.6
check "5 empty tasks (exit 2)"          2 "empty"                 --tasks "$empty"
check "6 below min_tasks (exit 2)"      2 "min_tasks"             --tasks "$tiny"

if [[ "$status" -eq 0 ]]; then
  echo "== all red cases failed the gate as required =="
else
  echo "== RED PROOF FAILED: a sabotage did not trip the gate ==" >&2
fi
exit "$status"

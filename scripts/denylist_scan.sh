#!/usr/bin/env bash
# Denylist scan: fail if any tracked file, or any commit in the full git history,
# matches a generic or private denylist pattern.
#
# Generic patterns: .denylist-generic.txt (committed, public).
# Private patterns: $AILAB_DENYLIST (newline-separated EREs), else the file at
#   ${AILAB_DENYLIST_FILE:-$HOME/.config/ailab/denylist.txt}. Never committed.
#
# Fails closed: with no private patterns the scan exits 2, unless
# AILAB_DENYLIST_OPTIONAL=1 (meant only for fork PRs, which cannot read secrets).
# Private pattern text is never printed; hits are reported by location and index.
#
# Exit codes: 0 clean, 1 hit found, 2 configuration error.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

generic_file=".denylist-generic.txt"
private_file="${AILAB_DENYLIST_FILE:-$HOME/.config/ailab/denylist.txt}"
commit_marker="ailab-denylist-commit "

die() {
  echo "denylist: ERROR: $*" >&2
  exit 2
}

# Print the EREs read from stdin, dropping blank and comment lines.
# (Written for bash 3.2, which macOS ships: no namerefs or mapfile.)
clean_patterns() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    printf '%s\n' "$line"
  done
}

generic=()
[[ -f "$generic_file" ]] || die "$generic_file is missing"
while IFS= read -r p; do generic+=("$p"); done < <(clean_patterns <"$generic_file")
((${#generic[@]} > 0)) || die "$generic_file has no patterns"

private=()
private_source=""
if [[ -n "${AILAB_DENYLIST:-}" ]]; then
  private_source="AILAB_DENYLIST"
  while IFS= read -r p; do private+=("$p"); done < <(printf '%s\n' "$AILAB_DENYLIST" | clean_patterns)
elif [[ -f "$private_file" ]]; then
  private_source="private denylist file"
  while IFS= read -r p; do private+=("$p"); done < <(clean_patterns <"$private_file")
fi

if ((${#private[@]} == 0)); then
  if [[ "${AILAB_DENYLIST_OPTIONAL:-0}" == "1" ]]; then
    echo "denylist: WARNING: no private patterns; AILAB_DENYLIST_OPTIONAL=1 so scanning generic patterns only." >&2
  else
    die "no private denylist patterns found. Set AILAB_DENYLIST (newline-separated EREs) or create ${private_file}. Refusing to pass without them (fail closed)."
  fi
else
  echo "denylist: loaded ${#generic[@]} generic and ${#private[@]} private patterns (from ${private_source})."
fi

# Validate every pattern up front: an invalid regex must fail, never silently match nothing.
check_regex() {
  local status=0
  grep -E -e "$1" </dev/null >/dev/null 2>&1 || status=$?
  ((status != 2)) || die "$2 is not a valid extended regex"
}
for i in "${!generic[@]}"; do check_regex "${generic[$i]}" "generic pattern #$((i + 1))"; done
if ((${#private[@]} > 0)); then
  for i in "${!private[@]}"; do check_regex "${private[$i]}" "private pattern #$((i + 1))"; done
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

# (a) Tracked files, excluding the generic list itself.
git ls-files -z -- . ":(exclude)$generic_file" >"$workdir/files"

# (b) Full history: every patch and commit message on every ref.
: >"$workdir/history"
if [[ -n "$(git rev-list --all -n 1)" ]]; then
  git log -p --all --diff-merges=separate --no-color --no-ext-diff \
    --format="${commit_marker}%H" -- . ":(exclude)$generic_file" >"$workdir/history"
fi

# Map a line number in the history dump to "<commit> <path>:<line>".
locate_history_line() {
  awk -v target="$1" -v marker="$commit_marker" '
    index($0, marker) == 1 { commit = substr($0, length(marker) + 1, 12); path = "(commit message)"; ln = 0 }
    /^\+\+\+ b\// { path = substr($0, 7) }
    /^@@ / { split($3, h, ","); ln = substr(h[1], 2) - 1 }
    /^[ +]/ && !/^\+\+\+ / { ln++ }
    NR == target { printf "%s %s:%s\n", commit, path, ln; exit }
  ' "$workdir/history"
}

hits=0

scan_pattern() {
  local pattern="$1" label="$2" loc lineno
  # Tracked files: grep prints file:line:content; keep only file:line.
  while IFS= read -r loc; do
    echo "denylist: HIT ${loc} matched ${label}"
    hits=$((hits + 1))
  done < <(xargs -0 -r grep -HnIiE -e "$pattern" -- <"$workdir/files" 2>/dev/null | cut -d: -f1,2 || true)
  # History: grep prints the line number in the dump; translate it to commit + path.
  while IFS= read -r lineno; do
    echo "denylist: HIT history $(locate_history_line "$lineno") matched ${label}"
    hits=$((hits + 1))
  done < <(grep -niE -e "$pattern" -- "$workdir/history" 2>/dev/null | cut -d: -f1 || true)
}

for i in "${!generic[@]}"; do
  scan_pattern "${generic[$i]}" "generic pattern #$((i + 1)) (${generic[$i]})"
done
if ((${#private[@]} > 0)); then
  for i in "${!private[@]}"; do
    scan_pattern "${private[$i]}" "private pattern #$((i + 1))"
  done
fi

file_count="$(tr -cd '\0' <"$workdir/files" | wc -c | tr -d ' ')"
commit_count="$(git rev-list --all | wc -l | tr -d ' ')"
if ((hits > 0)); then
  echo "denylist: FAIL: ${hits} hit(s) across ${file_count} tracked files and ${commit_count} commits." >&2
  exit 1
fi
echo "denylist: OK: ${file_count} tracked files and ${commit_count} commits are clean."

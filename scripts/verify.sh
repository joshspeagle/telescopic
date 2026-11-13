#!/usr/bin/env bash
# The gate: the commands of CI's lint and test jobs, run on this checkout's sources.
#   scripts/verify.sh [--cov] [--docs] [--slow]
# --cov adds the coverage floor of CI's coverage job, --docs the checks of the site's
# identity and the strict documentation build, and --slow the slow tests. CI installs the
# package; this runs it from src/.
# Set PYTHON to an interpreter that has the dev group (and the docs group for --docs)
# when that is not the first python on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

cov=""
docs=false
marker=()
for arg in "$@"; do
  case "$arg" in
    --cov) cov="--cov" ;;
    --docs) docs=true ;;
    --slow) marker=(-m "") ;;
    *) echo "usage: scripts/verify.sh [--cov] [--docs] [--slow]" >&2; exit 2 ;;
  esac
done

python="${PYTHON:-python}"
if ! "$python" -c "import numpy, scipy, pytest, ruff, mypy" 2>/dev/null; then
  echo "scripts/verify.sh: '$python' lacks the development dependencies." >&2
  echo "Install them with 'pip install -e . --group dev', or set PYTHON to an interpreter that has them." >&2
  exit 2
fi
# Absolute: the documentation build runs each notebook kernel in its page's folder.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$python" -m ruff check .
"$python" -m ruff format --check .
"$python" -m mypy
"$python" -m pytest -q -ra $cov "${marker[@]}"
if $docs; then
  "$python" docs/_scripts/brand.py --check
  "$python" docs/_scripts/hero_pair.py --check
  "$python" -m sphinx -W --keep-going docs docs/_build/html
fi

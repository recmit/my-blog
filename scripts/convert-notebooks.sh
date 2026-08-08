#!/usr/bin/env bash
#
# One-time conversion of notebooks/*.ipynb into src/content/posts/.
#
# This does NOT run in CI. It ran once and its output is committed; it is kept
# so the conversion can be corrected and repeated rather than hand-patched.
# Re-running it overwrites src/content/posts/ — including any front matter
# added by later phases, so check the diff.
#
# Notebooks are never executed: --to markdown reads the outputs already saved
# in the .ipynb files. Nothing here needs a 2022 Python environment, and
# neither pandas nor torch nor transformers is installed.
#
# Usage: scripts/convert-notebooks.sh

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$root/.venv-nbconvert"

if [ ! -d "$venv" ]; then
  echo "Creating $venv"
  python3 -m venv "$venv"
fi

# nbconvert pulls in nbformat; pyyaml reads the front matter out of _posts/.
"$venv/bin/pip" install --quiet --upgrade 'nbconvert~=7.16' 'pyyaml~=6.0'

echo "Converting notebooks:"
"$venv/bin/python" "$root/scripts/convert-notebooks.py"

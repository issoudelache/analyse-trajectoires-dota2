#!/usr/bin/env bash
# fac.sh — Lanceur pour les PCs de la fac (ajoute vendor/ au PYTHONPATH)
# Usage : ./fac.sh run.py compress --w_error 12
#         ./fac.sh gui.py
#         ./fac.sh benchmark/run_all.py --exp 0
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$DIR/vendor:$PYTHONPATH"
exec python3 "$@"

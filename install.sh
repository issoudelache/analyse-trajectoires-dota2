#!/usr/bin/env bash
# ============================================================
#  install.sh — Installation automatique (Linux / macOS)
# ============================================================
set -e

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERREUR : Python 3.10+ introuvable."
    echo "Ubuntu/Debian : sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "[1/3] Création de l'environnement virtuel..."
$PYTHON -m venv .venv

echo "[2/3] Activation du venv..."
source .venv/bin/activate

echo "[3/3] Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Installation terminée."
echo "Pour activer le venv : source .venv/bin/activate"

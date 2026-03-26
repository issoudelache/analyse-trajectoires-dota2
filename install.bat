@echo off
REM ============================================================
REM  install.bat — Installation automatique (Windows)
REM ============================================================

echo [1/3] Creation de l'environnement virtuel...
python -m venv .venv
if errorlevel 1 (
    echo ERREUR : Python 3.10+ introuvable. Installez Python depuis https://www.python.org
    exit /b 1
)

echo [2/3] Activation du venv...
call .venv\Scripts\activate.bat

echo [3/3] Installation des dependances...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Installation terminee.
echo Pour activer le venv : .venv\Scripts\activate

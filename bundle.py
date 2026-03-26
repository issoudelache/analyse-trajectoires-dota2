#!/usr/bin/env python3
"""
bundle.py — Télécharge les dépendances manquantes pour Ubuntu (PC fac) dans vendor/.
À exécuter sur votre machine (qui a pip) avant de copier le projet.

Usage : python bundle.py
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

VENDOR_DIR = "vendor"
TMP_DIR = "_tmp_wheels"

PACKAGES = [
    "pandas",
    "tqdm",
    "networkx",
    "scikit-learn",
    "customtkinter",
]

PYTHON_VERSION = "3.12"
PLATFORMS = [
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "linux_x86_64",
]


def main():
    shutil.rmtree(VENDOR_DIR, ignore_errors=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR)
    os.makedirs(VENDOR_DIR)

    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", TMP_DIR,
        "--python-version", PYTHON_VERSION,
        "--implementation", "cp",
        "--only-binary=:all:",
    ]
    for plat in PLATFORMS:
        cmd += ["--platform", plat]
    cmd += PACKAGES

    print("=== Telechargement des wheels Ubuntu (Python 3.12 x86_64) ===")
    subprocess.check_call(cmd)

    wheels = glob.glob(os.path.join(TMP_DIR, "*.whl"))
    print(f"\n=== Extraction de {len(wheels)} wheels dans {VENDOR_DIR}/ ===")
    for whl in sorted(wheels):
        print(f"  {os.path.basename(whl)}")
        with zipfile.ZipFile(whl, "r") as z:
            z.extractall(VENDOR_DIR)

    shutil.rmtree(TMP_DIR)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(VENDOR_DIR)
        for f in fns
    ) / (1024 * 1024)

    print(f"\n=== Termine ! vendor/ = {size_mb:.1f} Mo ===")
    print("Copiez tout le projet sur le PC de la fac.")
    print("Utilisez : ./fac.sh run.py compress --w_error 12")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Point d'entrée de l'interface graphique CustomTkinter.

Usage:
    python gui.py
"""

import sys
from pathlib import Path

# S'assurer que la racine du projet est dans le path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

from mvc.controllers.app_controller import AppController
from mvc.views.main_window import MainWindow


def main():
    controller = AppController()
    app = MainWindow(controller)
    app.mainloop()


if __name__ == "__main__":
    main()

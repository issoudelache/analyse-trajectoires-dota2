"""
BasePage — Page de base avec méthodes utilitaires.
"""

import customtkinter as ctk


class BasePage(ctk.CTkFrame):
    """Page de base avec méthodes utilitaires."""

    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller

    def on_show(self):
        """Appelé juste avant que la page soit affichée."""
        pass

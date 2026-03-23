"""
CompressPage — Page de lancement de compression.
"""

import customtkinter as ctk

from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, BG_CARD


class CompressPage(BasePage):
    """Page de lancement de compression."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self,
            text="Compression MDL",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=ACCENT,
        ).pack(pady=(30, 20))

        form = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        form.pack(padx=40, pady=10, fill="x")

        row1 = ctk.CTkFrame(form, fg_color="transparent")
        row1.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(row1, text="w_error:", width=100, anchor="w").pack(side="left")
        self.w_entry = ctk.CTkEntry(row1, width=120, placeholder_text="12.0")
        self.w_entry.pack(side="left", padx=10)
        self.w_entry.insert(0, "12.0")

        row2 = ctk.CTkFrame(form, fg_color="transparent")
        row2.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row2, text="Match ID:", width=100, anchor="w").pack(side="left")
        self.mid_entry = ctk.CTkEntry(row2, width=200, placeholder_text="(vide = tous)")
        self.mid_entry.pack(side="left", padx=10)

        self.run_btn = ctk.CTkButton(
            form,
            text="Lancer Compression",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_run,
        )
        self.run_btn.pack(pady=20)

        self.progress = ctk.CTkProgressBar(self, width=400)
        self.progress.pack(pady=10)
        self.progress.set(0)

        self.log_text = ctk.CTkTextbox(
            self, height=250, fg_color=BG_CARD, corner_radius=10
        )
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(5, 15))

    def _on_run(self):
        try:
            w = float(self.w_entry.get())
        except ValueError:
            return
        mid = self.mid_entry.get().strip() or None
        self.run_btn.configure(state="disabled", text="En cours…")
        self.log_text.delete("1.0", "end")
        self.progress.set(0)
        self.controller.start_compression(w, mid)

    def on_compress_progress(self, current, total, result):
        self.progress.set(current / max(total, 1))
        status = "OK" if result.success else f"ERREUR: {result.error}"
        line = f"[{current}/{total}] {result.match_id}: {status}"
        if result.success:
            line += f"  — {result.reduction_pct:.1f}% compression, {result.size_kb} KB"
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")

    def on_compress_done(self, results):
        self.run_btn.configure(state="normal", text="Lancer Compression")
        ok = sum(1 for r in results if r.success)
        self.log_text.insert(
            "end", f"\n{'=' * 50}\nTerminé: {ok}/{len(results)} matchs compressés.\n"
        )
        self.log_text.see("end")

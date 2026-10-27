import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk

import joblib
import pandas as pd

from src.database import Database
from src.fuzzy_matcher import get_candidates, normalize_name
from src.logger import get_logger
from src.ml.feature_extractor import extract_features
from src.pdf_processor import PDFProcessor
from src.web_bot import WebBot


logger = get_logger(__name__)


class CorrectionDialog(tk.Toplevel):
    def __init__(self, parent, pdf_name, candidates, preselect_name=None):
        super().__init__(parent)
        self.title("Potvrda korisnika")
        self.configure(bg="white")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None
        self.skipped = False

        self.selected_var = tk.StringVar(value=preselect_name or "")
        self.manual_var = tk.StringVar(value="")

        container = tk.Frame(self, bg="white", padx=14, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            container,
            text=f"PDF ime: {pdf_name}",
            bg="white",
            fg="#2c3e50",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))

        if candidates:
            tk.Label(
                container,
                text="Predlozeni kandidati:",
                bg="white",
                fg="#34495e",
                font=("Segoe UI", 10),
                anchor="w",
            ).pack(fill=tk.X, pady=(0, 4))

        rb_wrap = tk.Frame(container, bg="white")
        rb_wrap.pack(fill=tk.X, pady=(0, 8))

        for cand in candidates:
            score = cand.get("ml_score")
            fuzzy = cand.get("f_wratio", 0)
            if score is None:
                line = f"[--] {cand['name']}  (fuzzy: {fuzzy:.0f})"
            else:
                line = f"[{score:.0%}] {cand['name']}  (fuzzy: {fuzzy:.0f})"
            tk.Radiobutton(
                rb_wrap,
                text=line,
                value=cand["name"],
                variable=self.selected_var,
                bg="white",
                anchor="w",
                justify=tk.LEFT,
                activebackground="white",
                selectcolor="white",
            ).pack(fill=tk.X, anchor="w")

        tk.Label(
            container,
            text="Ili rucni unos:",
            bg="white",
            fg="#34495e",
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill=tk.X, pady=(6, 2))

        manual_entry = tk.Entry(container, textvariable=self.manual_var, width=44)
        manual_entry.pack(fill=tk.X, pady=(0, 10))
        manual_entry.focus_set()

        btn_row = tk.Frame(container, bg="white")
        btn_row.pack(fill=tk.X)

        tk.Button(
            btn_row,
            text="Potvrdi",
            bg="#16a085",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._confirm,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row,
            text="Preskoci",
            bg="#f39c12",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._skip,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_row,
            text="Zatvori",
            bg="#7f8c8d",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._close,
            padx=10,
            pady=4,
        ).pack(side=tk.RIGHT)

        self.bind("<Return>", lambda _e: self._confirm())
        self.bind("<Escape>", lambda _e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._skip)

    def _confirm(self):
        manual_text = (self.manual_var.get() or "").strip()
        selected = (self.selected_var.get() or "").strip()
        final_name = manual_text or selected
        if not final_name:
            messagebox.showwarning("Unos potreban", "Odaberite kandidata ili unesite ime.")
            return
        self.result = final_name
        self.destroy()

    def _skip(self):
        self.skipped = True
        self.result = None
        self.destroy()

    def _close(self):
        self.result = None
        self.destroy()


class AutomationApp:
    DEFAULT_MODEL_PATH = os.path.join(
        "models",
        "EXP2_fuzzy_plus_engineering__RandomForest.pkl",
    )

    def __init__(self, root):
        self.root = root
        self.root.title("Web Automation - Unos placanja")
        self.root.geometry("980x760")
        self.root.configure(bg="#ecf0f1")

        self.db = Database()
        self.pdf_processor = PDFProcessor()

        self.pdf_path = tk.StringVar(value="")
        self.model_path = tk.StringVar(value=self.DEFAULT_MODEL_PATH)
        self.model_status_var = tk.StringVar(value="❌ Nije ucitan")
        self.status_var = tk.StringVar(value="Spremno.")

        self.df = None
        self.branch = ""
        self.start_page = tk.IntVar(value=1)

        self.ml_model = None
        self.HIGH_CONFIDENCE = 0.80
        self.LOW_CONFIDENCE = 0.50
        self.ml_stats = {"auto": 0, "confirmed": 0, "manual": 0, "unresolved": 0}
        self.feature_columns = [
            "f_ratio",
            "f_partial",
            "f_token_sort",
            "f_token_set",
            "f_wratio",
            "len_diff",
            "len_ratio",
            "token_count_diff",
            "jaccard",
            "first_token_sim",
            "shared_fuzzy_tokens",
            "jaro_winkler",
            "candidate_rank",
        ]

        self._is_running = False
        self._is_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._worker_thread = None
        self.correction_queue = queue.Queue()

        self._build_ui()
        self._load_default_model()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#2c3e50", height=56)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Automatizacija unosa placanja",
            bg="#2c3e50",
            fg="white",
            font=("Segoe UI", 14, "bold"),
            pady=12,
        ).pack(anchor="w", padx=14)

        body = tk.Frame(self.root, bg="#ecf0f1")
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        step1 = tk.LabelFrame(body, text="Step 1 - Odabir PDF fajla", bg="#ecf0f1", fg="#2c3e50")
        step1.pack(fill=tk.X, pady=(0, 8))
        row1 = tk.Frame(step1, bg="#ecf0f1")
        row1.pack(fill=tk.X, padx=0, pady=(8, 0))
        tk.Entry(row1, textvariable=self.pdf_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=0)
        tk.Button(
            row1,
            text="Odaberi PDF",
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.select_pdf,
            padx=12,
            pady=4,
        ).pack(side=tk.RIGHT, padx=8, pady=0)

        row1b = tk.Frame(step1, bg="#ecf0f1")
        row1b.pack(fill=tk.X, padx=8, pady=(6, 10))
        tk.Label(
            row1b,
            text="Pocni upis od stranice:",
            bg="#ecf0f1",
            fg="#2c3e50",
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT)
        tk.Spinbox(
            row1b,
            from_=1,
            to=9999,
            width=6,
            textvariable=self.start_page,
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=(6, 4))
        tk.Label(
            row1b,
            text="(1 = prva stranica PDF-a)",
            bg="#ecf0f1",
            fg="#7f8c8d",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        ml_frame = tk.LabelFrame(body, text="🧠 ML Model", bg="#ecf0f1", fg="#2c3e50")
        ml_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Entry(ml_frame, textvariable=self.model_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=10)
        tk.Label(
            ml_frame,
            textvariable=self.model_status_var,
            bg="#ecf0f1",
            fg="#2c3e50",
            font=("Segoe UI", 10, "bold"),
            width=20,
            anchor="w",
        ).pack(side=tk.RIGHT, padx=8, pady=10)

        step2 = tk.LabelFrame(body, text="Step 2 - Preview", bg="#ecf0f1", fg="#2c3e50")
        step2.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        preview_top = tk.Frame(step2, bg="#ecf0f1")
        preview_top.pack(fill=tk.X, padx=8, pady=(8, 2))
        tk.Button(
            preview_top,
            text="Prikazi Preview",
            bg="#16a085",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.show_preview,
            padx=12,
            pady=4,
        ).pack(side=tk.LEFT)

        self.preview_text = scrolledtext.ScrolledText(step2, height=20, wrap=tk.NONE)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 10))

        step3 = tk.LabelFrame(body, text="Step 3 - Automatizacija", bg="#ecf0f1", fg="#2c3e50")
        step3.pack(fill=tk.X)

        btn_row = tk.Frame(step3, bg="#ecf0f1")
        btn_row.pack(fill=tk.X, padx=8, pady=(10, 6))
        tk.Button(
            btn_row,
            text="Start",
            bg="#27ae60",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_automation,
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn_row,
            text="Pauza / Nastavi",
            bg="#f39c12",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.toggle_pause,
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn_row,
            text="Stop",
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.stop_automation,
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn_row,
            text="Zavrsi i Snimi",
            bg="#16a085",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.finish_and_save,
            padx=14,
            pady=4,
        ).pack(side=tk.RIGHT)

        self.progress = ttk.Progressbar(step3, mode="determinate")
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 8))

        status_row = tk.Frame(step3, bg="#ecf0f1")
        status_row.pack(fill=tk.X, padx=8, pady=(0, 10))
        tk.Label(
            status_row,
            textvariable=self.status_var,
            bg="#ecf0f1",
            fg="#2c3e50",
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill=tk.X)

    def _ui(self, callback, *args, **kwargs):
        self.root.after(0, lambda: callback(*args, **kwargs))

    def _get_start_page(self) -> int:
        """1-based PDF page from which extraction and entry begin."""
        try:
            return max(1, int(self.start_page.get()))
        except (tk.TclError, ValueError, TypeError):
            return 1

    def select_pdf(self):
        path = filedialog.askopenfilename(
            title="Odaberi PDF",
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        self.pdf_path.set(path)
        self.status_var.set("PDF odabran. Pokrenite preview.")

    def _load_default_model(self):
        model_path = self.DEFAULT_MODEL_PATH
        self.model_path.set(model_path)
        self._load_model(model_path, show_popup=False)

    def _load_model(self, path, show_popup=True):
        try:
            model = joblib.load(path)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model nema predict_proba metodu.")
            self.ml_model = model
            self.model_status_var.set(f"✅ {os.path.splitext(os.path.basename(path))[0]}")
            self.status_var.set("ML model uspjesno ucitan.")
        except Exception as exc:
            self.ml_model = None
            self.model_status_var.set("❌ Nije ucitan")
            self.status_var.set("RandomForest EXP2 model nije ucitan. Koristi se fuzzy-only fallback.")
            if show_popup:
                messagebox.showerror("Greska modela", f"Ne mogu ucitati model:\n{exc}")
            logger.exception("Failed loading model: %s", exc)

    def _extract_feature_vector(self, features):
        if self.ml_model is None:
            return None, False

        use_embeddings = False
        feature_names = None
        if hasattr(self.ml_model, "feature_names_in_"):
            feature_names = list(self.ml_model.feature_names_in_)
            use_embeddings = "cosine_sim" in feature_names
        elif hasattr(self.ml_model, "named_steps"):
            final_estimator = list(self.ml_model.named_steps.values())[-1]
            if hasattr(final_estimator, "feature_names_in_"):
                feature_names = list(final_estimator.feature_names_in_)
                use_embeddings = "cosine_sim" in feature_names

        if feature_names is None:
            feature_names = sorted(features.keys())
            use_embeddings = "cosine_sim" in feature_names

        vec = {name: features.get(name, 0.0) for name in feature_names}
        frame = pd.DataFrame([vec], columns=feature_names)
        return frame, use_embeddings

    def _ml_decide(self, pdf_name, candidates):
        """
        Returns (best_name, best_score, decision, scored_candidates).
        decision: "auto" | "uncertain" | "no_model"
        """
        if not candidates:
            return "", 0.0, "uncertain", []

        if self.ml_model is None:
            fallback = []
            for cand in candidates:
                cand_copy = dict(cand)
                cand_copy["ml_score"] = 0.0
                fallback.append(cand_copy)
            return candidates[0]["name"], 0.0, "no_model", fallback

        pdf_norm = normalize_name(pdf_name)
        scored = []
        for rank, cand in enumerate(candidates, start=1):
            features = extract_features(
                pdf_name=pdf_name,
                pdf_name_norm=pdf_norm,
                candidate=cand,
                candidate_rank=rank,
                use_embeddings=False,
            )
            try:
                feature_names = None
                if hasattr(self.ml_model, "feature_names_in_"):
                    feature_names = list(self.ml_model.feature_names_in_)
                else:
                    feature_names = list(self.feature_columns)

                row = {col: features.get(col, 0.0) for col in feature_names}
                frame = pd.DataFrame([row], columns=feature_names)
                prob = float(self.ml_model.predict_proba(frame)[0][1])
            except Exception as exc:
                logger.warning("ML scoring failed, fallback to fuzzy ranking: %s", exc)
                prob = float(cand.get("f_wratio", 0.0) / 100.0)
            cand_copy = dict(cand)
            cand_copy["ml_score"] = prob
            scored.append(cand_copy)

        scored.sort(key=lambda c: c.get("ml_score", 0.0), reverse=True)
        best = scored[0]
        best_name = best["name"]
        best_score = float(best.get("ml_score", 0.0))
        decision = "auto" if best_score >= self.HIGH_CONFIDENCE else "uncertain"
        return best_name, best_score, decision, scored

    def show_preview(self):
        if not self.pdf_path.get():
            messagebox.showwarning("Nedostaje PDF", "Odaberite PDF fajl.")
            return
        if self._is_running:
            messagebox.showinfo("U toku", "Automatizacija je vec pokrenuta.")
            return

        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, "Generisem preview, sacekajte...\n")
        self.status_var.set("Priprema preview-a...")

        worker = threading.Thread(target=self._build_preview_worker, daemon=True)
        worker.start()

    def _build_preview_worker(self):
        try:
            df = self.pdf_processor.extract_payments(
                self.pdf_path.get(),
                start_page=self._get_start_page(),
            )
            branch = self.pdf_processor._extract_branch(self.pdf_path.get())
            logger.info("Filijala iz zaglavlja PDF-a: %r", branch)

            lines = []
            for _, row in df.iterrows():
                pdf_name = str(row["ime_prezime"]).strip()
                amount = str(row["rata"]).strip()
                lines.append(f"{pdf_name}\t{amount}\n")

            rates = pd.to_numeric(
                df["rata"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            ).fillna(0.0)
            total = float(rates.sum())

            lines.append("\n")
            lines.append(f"UKUPNO RATA: {total:.2f}\n")

            preview_text = "".join(lines)

            def _apply():
                self.df = df
                self.branch = branch
                self.preview_text.delete("1.0", tk.END)
                self.preview_text.insert(tk.END, preview_text)
                self.status_var.set(f"Preview: {len(df)} redova, ukupno rata {total:.2f}")

            self._ui(_apply)
        except Exception as exc:
            logger.exception("Preview build failed: %s", exc)
            self._ui(messagebox.showerror, "Greska", f"Neuspjesan preview:\n{exc}")
            self._ui(self.status_var.set, "Preview nije uspio.")

    def start_automation(self):
        if self._is_running:
            return
        if not self.pdf_path.get():
            messagebox.showwarning("Nedostaje PDF", "Odaberite PDF fajl.")
            return
        if self.df is None:
            try:
                self.df = self.pdf_processor.extract_payments(
                    self.pdf_path.get(),
                    start_page=self._get_start_page(),
                )
                # Filijala iz zaglavlja PDF-a — koristi se kao bonus pri rangiranju (ne filter)
                self.branch = self.pdf_processor._extract_branch(self.pdf_path.get())
                logger.info("Filijala iz zaglavlja PDF-a: %r", self.branch)
            except Exception as exc:
                messagebox.showerror("Greska", f"Ne mogu procitati PDF:\n{exc}")
                return
        if self.df.empty:
            messagebox.showwarning("Prazan PDF", "Nema redova za unos.")
            return

        self.ml_stats = {"auto": 0, "confirmed": 0, "manual": 0, "unresolved": 0}
        self._is_running = True
        self._is_paused = False
        self._pause_event.set()

        self.progress["maximum"] = len(self.df)
        self.progress["value"] = 0
        self.status_var.set("Pokrecem browser i automatizaciju...")

        self._worker_thread = threading.Thread(target=self._run_automation, daemon=True)
        self._worker_thread.start()

    def toggle_pause(self):
        if not self._is_running:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_event.clear()
            self.status_var.set("Pauzirano.")
        else:
            self._pause_event.set()
            self.status_var.set("Nastavljeno.")

    def stop_automation(self):
        if not self._is_running:
            return
        self._is_running = False
        self._pause_event.set()
        self.status_var.set("Zaustavljanje...")

    def _run_automation(self):
        stats = {
            "total": len(self.df),
            "direct": 0,
            "ml_auto": 0,
            "ml_confirmed": 0,
            "manual": 0,
            "unresolved": 0,
        }

        bot = None
        table_ready = False
        try:
            bot = WebBot()
            bot.start()
            bot.login()
            bot.navigate_to_table()
            table_ready = True

            for idx, (_, row) in enumerate(self.df.iterrows(), start=1):
                if not self._is_running:
                    break
                self._pause_event.wait()
                if not self._is_running:
                    break

                pdf_name = str(row["ime_prezime"]).strip()
                amount = str(row["rata"]).strip()

                alias = self.db.get_alias(pdf_name)
                name_to_try = alias["web_name"] if alias else pdf_name

                self._ui(self.status_var.set, f"[{idx}/{stats['total']}] Unosem: {pdf_name} -> pokusavam direktno...")
                success = bot.enter_payment(name_to_try, amount)

                if success:
                    stats["direct"] += 1
                    self._ui(self.status_var.set, f"[{idx}/{stats['total']}] Unosem: {pdf_name} -> ✅ Direktno")
                    self._ui(self.progress.configure, value=idx)
                    continue

                self._ui(self.status_var.set, f"[{idx}/{stats['total']}] Unosem: {pdf_name} -> ❌ Nema autocomplete, fallback...")

                candidates = get_candidates(pdf_name, self.branch, top_n=5)
                best_name, best_score, decision, scored_candidates = self._ml_decide(pdf_name, candidates)

                if decision == "auto":
                    self._ui(self.status_var.set, f"[{idx}/{stats['total']}] Unosem: {pdf_name} -> 🤖 ML auto-fix ({best_score:.0%})")
                    success2 = bot.retry_name_in_current_row(best_name, amount)
                    if success2:
                        self.db.save_alias(
                            pdf_name=pdf_name,
                            web_name=best_name,
                            confidence=best_score,
                            confirmed=True,
                            source="ml_auto",
                            filijala=self.branch,
                        )
                        stats["ml_auto"] += 1
                        self.ml_stats["auto"] += 1
                    else:
                        self.db.add_unresolved(pdf_name, notes="ML auto-fix failed in web form")
                        stats["unresolved"] += 1
                        self.ml_stats["unresolved"] += 1
                    self._ui(self.progress.configure, value=idx)
                    continue

                preselect = best_name if best_name else None
                self._ui(self.status_var.set, f"[{idx}/{stats['total']}] Unosem: {pdf_name} -> 🔍 Cekam potvrdu...")
                answer = self._ask_for_correction(
                    pdf_name=pdf_name,
                    candidates=scored_candidates,
                    preselect_name=preselect,
                )

                if answer["skipped"] or not answer["name"]:
                    bot._delete_current_empty_row()
                    self.db.add_unresolved(pdf_name, notes="Skipped by user")
                    stats["unresolved"] += 1
                    self.ml_stats["unresolved"] += 1
                    self._ui(self.progress.configure, value=idx)
                    continue

                chosen_name = answer["name"]
                success3 = bot.retry_name_in_current_row(chosen_name, amount)
                if not success3:
                    self.db.add_unresolved(pdf_name, notes="Manual/confirmed entry failed in web form")
                    stats["unresolved"] += 1
                    self.ml_stats["unresolved"] += 1
                    self._ui(self.progress.configure, value=idx)
                    continue

                source = "ml_confirmed" if decision == "uncertain" and self.ml_model is not None else "manual"
                self.db.save_alias(
                    pdf_name=pdf_name,
                    web_name=chosen_name,
                    confidence=best_score if source == "ml_confirmed" else 0.0,
                    confirmed=True,
                    source=source,
                    filijala=self.branch,
                )
                if source == "ml_confirmed":
                    stats["ml_confirmed"] += 1
                    self.ml_stats["confirmed"] += 1
                else:
                    stats["manual"] += 1
                    self.ml_stats["manual"] += 1
                self._ui(self.progress.configure, value=idx)

            self._ui(self._show_results, stats)
        except Exception as exc:
            logger.exception("Automation failed: %s", exc)
            self._ui(messagebox.showerror, "Greska automatizacije", str(exc))
            self._ui(self.status_var.set, "Automatizacija prekinuta zbog greske.")
        finally:
            self._is_running = False
            self._pause_event.set()
            if bot is not None:
                try:
                    if table_ready:
                        bot.save()
                except Exception as save_exc:
                    logger.warning("Save prije zatvaranja browsera nije uspio: %s", save_exc)
                try:
                    bot.close()
                except Exception:
                    pass

    def _ask_for_correction(self, pdf_name, candidates, preselect_name=None):
        while True:
            try:
                self.correction_queue.get_nowait()
            except queue.Empty:
                break

        self.root.after(
            0,
            lambda: self._show_correction_dialog(
                pdf_name=pdf_name,
                candidates=candidates,
                preselect_name=preselect_name,
            ),
        )
        try:
            return self.correction_queue.get(timeout=300)
        except queue.Empty:
            logger.warning("CorrectionDialog timeout for: %s", pdf_name)
            return {"name": None, "skipped": True}

    def _show_correction_dialog(self, pdf_name, candidates, preselect_name=None):
        result = {"name": None, "skipped": True}
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            dialog = CorrectionDialog(
                parent=self.root,
                pdf_name=pdf_name,
                candidates=candidates,
                preselect_name=preselect_name,
            )
            dialog.update_idletasks()
            dialog.lift()
            dialog.attributes("-topmost", True)
            dialog.after(300, lambda: dialog.attributes("-topmost", False))
            self.root.wait_window(dialog)
            result = {"name": dialog.result, "skipped": dialog.skipped}
        except Exception as exc:
            logger.error("CorrectionDialog error for %s: %s", pdf_name, exc)
            result = {"name": None, "skipped": True}
        finally:
            self.correction_queue.put(result)

    def _show_results(self, stats):
        total = max(stats["total"], 1)
        direct_pct = stats["direct"] / total * 100.0
        auto_pct = stats["ml_auto"] / total * 100.0
        conf_pct = stats["ml_confirmed"] / total * 100.0
        man_pct = stats["manual"] / total * 100.0
        unr_pct = stats["unresolved"] / total * 100.0

        text = (
            "Zavrseno!\n\n"
            f"Ukupno:              {stats['total']}\n"
            f"✅ Direktno (bot):   {stats['direct']}  ({direct_pct:5.1f}%)\n"
            f"🤖 ML auto-fix:      {stats['ml_auto']}  ({auto_pct:5.1f}%)\n"
            f"🔍 ML + potvrda:     {stats['ml_confirmed']}  ({conf_pct:5.1f}%)\n"
            f"✏️  Rucno:            {stats['manual']}  ({man_pct:5.1f}%)\n"
            f"⏭️  Preskoceno:       {stats['unresolved']}  ({unr_pct:5.1f}%)\n"
        )
        self.status_var.set("Automatizacija zavrsena.")
        messagebox.showinfo("Rezultat", text)

    def finish_and_save(self):
        if self._is_running:
            messagebox.showinfo("U toku", "Automatizacija je jos u toku.")
            return
        self.status_var.set("Sve izmjene su obradjene.")
        messagebox.showinfo("Info", "Ako je unos zavrsen, podaci su vec snimljeni klikom na Save u web aplikaciji.")


def main():
    root = tk.Tk()
    app = AutomationApp(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()

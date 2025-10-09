import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# moduly
from klasyfikacja import (
    ucz_model_bovw_svm,
    ocen_klasyfikator,
    zapisz_modeli_slownik,
    wczytaj_modeli_slownik,
    przewidz_etykiete
)

# Stale sciezki
PROJEKT = Path(__file__).resolve().parent.parent
F_DANE = PROJEKT / "dane"
F_TRAIN = F_DANE / "trening"
F_TEST = F_DANE / "test"
F_WYNIKI = PROJEKT / "wyniki"

class AplikacjaOCR(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("System klasyfikacji dokumentów z ekstrakcją kluczowych informacji")
        self.geometry("980x680")

        # Stan aplikacji
        self.model = None
        self.slownik = None
        self.id2nazwa = None
        self.obraz_testowy_path: Path | None = None
        self.predykcja_label: str | None = None
        self.trwa_uczenie = False

        # UI
        self._zbuduj_menu()
        self._zbuduj_gorny_panel()
        self._zbuduj_srodkowy_panel()
        self._zbuduj_dolny_panel()

        # Przyciski startowe
        self._ustaw_stan_poczatkowy()

        # katalog wyniki istnieje ?
        F_WYNIKI.mkdir(parents=True, exist_ok=True)

    # ---------- Layout ----------
    def _zbuduj_menu(self):
        menubar = tk.Menu(self)
        plik_menu = tk.Menu(menubar, tearoff=0)
        plik_menu.add_command(label="Wczytaj zapisany model…", command=self._akcja_wczytaj_model)
        plik_menu.add_command(label="Zapisz aktualny model", command=self._akcja_zapisz_model)
        plik_menu.add_separator()
        plik_menu.add_command(label="Wyjście", command=self.destroy)
        menubar.add_cascade(label="Plik", menu=plik_menu)

        pomoc_menu = tk.Menu(menubar, tearoff=0)
        pomoc_menu.add_command(label="Informacje o projekcie", command=lambda: messagebox.showinfo(
            "Projekt \n" ,
            "System klasyfikacji dokumentów z ekstrakcją kluczowych informacji. \n"
            "\nAutorzy: \n"
            "- Tomasz Chojnacki \n"
            "- Maciej Bernatek \n"
            "- Przemysław Pałka \n"
            "Grupa 4ID14A \n"
        ))
        menubar.add_cascade(label="Pomoc", menu=pomoc_menu)
        self.config(menu=menubar)

    def _zbuduj_gorny_panel(self):
        ramka = ttk.LabelFrame(self, text="Uczenie i ewaluacja")
        ramka.pack(fill="x", padx=10, pady=8)

        self.btn_ucz = ttk.Button(ramka, text="Rozpocznij proces uczenia", command=self._akcja_uczenie)
        self.btn_ucz.grid(row=0, column=0, padx=6, pady=6, sticky="w")

        ttk.Label(ramka, text="k (słownik BoVW):").grid(row=0, column=1, padx=(18,4), sticky="e")
        self.var_k = tk.IntVar(value=300)
        self.ent_k = ttk.Entry(ramka, width=6, textvariable=self.var_k)
        self.ent_k.grid(row=0, column=2, padx=4)

        ttk.Label(ramka, text="C (SVM):").grid(row=0, column=3, padx=(18,4), sticky="e")
        self.var_C = tk.DoubleVar(value=2.0)
        self.ent_C = ttk.Entry(ramka, width=6, textvariable=self.var_C)
        self.ent_C.grid(row=0, column=4, padx=4)

        self.btn_wczytaj_model = ttk.Button(ramka, text="Wczytaj zapisany model", command=self._akcja_wczytaj_model)
        self.btn_wczytaj_model.grid(row=0, column=5, padx=10, pady=6)

        self.prog = ttk.Progressbar(ramka, mode="indeterminate", length=240)
        self.prog.grid(row=0, column=6, padx=10, pady=6)

        self.lbl_status = ttk.Label(ramka, text="Status: gotowy")
        self.lbl_status.grid(row=1, column=0, columnspan=7, sticky="w", padx=6, pady=(0,6))

    def _zbuduj_srodkowy_panel(self):
        kontener = ttk.Frame(self)
        kontener.pack(fill="both", expand=True, padx=10, pady=4)

        # Lewa kolumna: wybór i podgląd obrazu
        lewa = ttk.LabelFrame(kontener, text="Obraz do rozpoznania")
        lewa.pack(side="left", fill="both", expand=True, padx=(0,6))

        self.btn_wybierz = ttk.Button(lewa, text="Wybierz obraz…", command=self._akcja_wybierz_obraz)
        self.btn_wybierz.pack(anchor="w", padx=8, pady=8)

        self.lbl_podglad = ttk.Label(lewa)
        self.lbl_podglad.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # Prawa kolumna: działania i wynik identyfikacji
        prawa = ttk.LabelFrame(kontener, text="Identyfikacja")
        prawa.pack(side="left", fill="both", expand=True, padx=(6,0))

        self.btn_rozpoznaj = ttk.Button(prawa, text="Wyświetl identyfikację", command=self._akcja_rozpoznaj)
        self.btn_rozpoznaj.pack(anchor="w", padx=8, pady=(8,2))

        self.lbl_wynik = ttk.Label(prawa, text="Brak wyniku", font=("Segoe UI", 12, "bold"))
        self.lbl_wynik.pack(anchor="w", padx=8, pady=(2,8))

        ttk.Separator(prawa, orient="horizontal").pack(fill="x", padx=8, pady=4)

        self.btn_zapisz_model = ttk.Button(prawa, text="Zapisz aktualny model", command=self._akcja_zapisz_model)
        self.btn_zapisz_model.pack(anchor="w", padx=8, pady=4)

        self.btn_wyczysc_podglad = ttk.Button(prawa, text="Wyczyść podgląd", command=self._akcja_wyczysc_podglad)
        self.btn_wyczysc_podglad.pack(anchor="w", padx=8, pady=4)

    def _zbuduj_dolny_panel(self):
        ramka = ttk.LabelFrame(self, text="Log")
        ramka.pack(fill="both", expand=False, padx=10, pady=(4,10))
        self.txt_log = tk.Text(ramka, height=10)
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)
        self.btn_wyczysc_log = ttk.Button(ramka, text="Wyczyść log", command=lambda: self.txt_log.delete("1.0", "end"))
        self.btn_wyczysc_log.pack(anchor="e", padx=8, pady=(0,6))

    # ---------- Stany przycisków ----------
    def _ustaw_stan_poczatkowy(self):
        self._ustaw_status("Status: gotowy")
        self._ustaw_trwa_uczenie(False)
        self._zablokuj(self.btn_rozpoznaj, True)  # rozpoznaj – na razie nie
        self._zablokuj(self.btn_wybierz, True)    # wybór obrazu dopiero po modelu
        self._zablokuj(self.btn_zapisz_model, True)

    def _po_uczeniu_aktywuj(self):
        self._zablokuj(self.btn_wybierz, False)
        self._zablokuj(self.btn_zapisz_model, False)

    def _zablokuj(self, widg, blokuj: bool):
        state = "disabled" if blokuj else "normal"
        widg.config(state=state)

    def _ustaw_trwa_uczenie(self, trwa: bool):
        self.trwa_uczenie = trwa
        if trwa:
            self.prog.start(10)
            self._zablokuj(self.btn_ucz, True)
            self._zablokuj(self.btn_wczytaj_model, True)
        else:
            self.prog.stop()
            self._zablokuj(self.btn_ucz, False)
            self._zablokuj(self.btn_wczytaj_model, False)

    def _ustaw_status(self, txt: str):
        self.lbl_status.config(text=txt)
        self._log(txt)

    def _log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    # ---------- Akcje ----------
    def _akcja_uczenie(self):
        # Walidacja katalogów
        if not F_TRAIN.exists() or not F_TEST.exists():
            messagebox.showerror("Błąd", f"Brak folderów danych:\n{F_TRAIN}\n{F_TEST}")
            return
        try:
            k = int(self.var_k.get())
            C = float(self.var_C.get())
        except Exception:
            messagebox.showerror("Błąd", "Parametry k i C muszą być liczbami.")
            return

        def worker():
            try:
                self._ustaw_trwa_uczenie(True)
                self._ustaw_status("Uczenie w toku…")
                self.model, self.slownik, self.id2nazwa = ucz_model_bovw_svm(F_TRAIN, k=k, C=C)
                self._ustaw_status("Uczenie zakończone. Ocena na zbiorze testowym…")
                acc, f1, _ = ocen_klasyfikator(self.model, self.slownik, F_TEST, self.id2nazwa, F_WYNIKI / "macierz_pomylek.png")
                self._ustaw_status(f"Accuracy: {acc:.3f} | F1 (macro): {f1:.3f}")
                self._po_uczeniu_aktywuj()

            except Exception as e:
                messagebox.showerror("Błąd podczas uczenia", str(e))
                self._log(f"[BŁĄD] {e}")
            finally:
                self._ustaw_trwa_uczenie(False)
                self._ustaw_status("Gotowe.")

        threading.Thread(target=worker, daemon=True).start()

    def _akcja_wczytaj_model(self):
        try:
            # Obsłuż obie wersje funkcji (z mapą lub bez)
            wynik = wczytaj_modeli_slownik(F_WYNIKI)
            if isinstance(wynik, tuple) and len(wynik) == 3:
                self.model, self.slownik, self.id2nazwa = wynik
            else:
                self.model, self.slownik = wynik  # stara wersja – bez mapy
                # jeśli brak mapy, odtwórz z folderów treningowych
                _, _, self.id2nazwa = self._odczytaj_mapowanie_z_dysku()
            self._ustaw_status("Wczytano zapisany model.")
            self._po_uczeniu_aktywuj()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać modelu:\n{e}")
            self._log(f"[BŁĄD] {e}")

    def _akcja_zapisz_model(self):
        if not self.model or not self.slownik:
            messagebox.showwarning("Uwaga", "Brak wytrenowanego modelu do zapisania.")
            return
        try:
            if self.id2nazwa is None:
                _, _, self.id2nazwa = self._odczytaj_mapowanie_z_dysku()
            zapisz_modeli_slownik(self.model, self.slownik, self.id2nazwa, F_WYNIKI)
            self._ustaw_status(f"Zapisano model i słownik do: {F_WYNIKI}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać modelu:\n{e}")
            self._log(f"[BŁĄD] {e}")

    def _odczytaj_mapowanie_z_dysku(self):
        # Buduje mapę id->nazwa po alfabecie folderów w treningu (spójnie z ucz_model_bovw_svm)
        podf = sorted([p for p in F_TRAIN.iterdir() if p.is_dir()])
        id2nazwa = {i: p.name for i, p in enumerate(podf)}
        sciezki = []
        y = []
        for i, p in enumerate(podf):
            for f in p.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    sciezki.append(f)
                    y.append(i)
        return sciezki, y, id2nazwa

    def _akcja_wybierz_obraz(self):
        if not self.model or not self.slownik:
            messagebox.showwarning("Uwaga", "Najpierw wytrenuj lub wczytaj model.")
            return
        plik = filedialog.askopenfilename(
            title="Wybierz obraz dokumentu",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not plik:
            return
        self.obraz_testowy_path = Path(plik)
        self._wczytaj_podglad(self.obraz_testowy_path)
        self._ustaw_status(f"Wybrany obraz: {self.obraz_testowy_path.name}")
        self._zablokuj(self.btn_rozpoznaj, False)

    def _akcja_wyczysc_podglad(self):
        self.obraz_testowy_path = None
        self.lbl_podglad.config(image="", text="")
        self.lbl_wynik.config(text="Brak wyniku")
        self._zablokuj(self.btn_rozpoznaj, True)

    def _akcja_rozpoznaj(self):
        if not (self.model and self.slownik and self.obraz_testowy_path):
            messagebox.showwarning("Uwaga", "Brak modelu lub obrazu.")
            return
        try:
            etyk_num = przewidz_etykiete(self.model, self.slownik, self.obraz_testowy_path)
            nazwa = self.id2nazwa[int(etyk_num)] if self.id2nazwa else str(etyk_num)
            self.lbl_wynik.config(text=f"Rozpoznano jako: {nazwa.upper()}")
            self._ustaw_status(f"Wynik: {nazwa}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się rozpoznać obrazu:\n{e}")
            self._log(f"[BŁĄD] {e}")

    # ---------- Pomocnicze ----------
    def _wczytaj_podglad(self, sciezka: Path, max_w=440, max_h=360):
        img = Image.open(sciezka).convert("RGB")
        # dopasowanie z zachowaniem proporcji
        img.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(img)  # zachowaj referencję!
        self.lbl_podglad.config(image=self._photo)

if __name__ == "__main__":
    app = AplikacjaOCR()
    app.mainloop()

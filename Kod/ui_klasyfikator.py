import queue
import threading

from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from klasyfikacja import (
    ucz_model_bovw_svm,
    ocen_klasyfikator,
    zapisz_modeli_slownik,
    wczytaj_modeli_slownik,
    przewidz_etykiete
)

from ocr import (ocr_tekst_z_obrazu,ocr_tekst_z_biletu)

KATALOG_KODU = Path(__file__).resolve().parent # Pliki aplikacji znajdują się w folderze Kod

PROJEKT = KATALOG_KODU.parent # Folder główny projektu

F_DANE = PROJEKT / "dane"
F_TRAIN = F_DANE / "trening"
F_TEST = F_DANE / "test"

F_WYNIKI = PROJEKT / "wyniki"
F_ODCZYTANE = PROJEKT / "odczytane"

# KLASA INTERFEJSU KLASYFIKATORA
class WidokKlasyfikatora(ttk.Frame):

    def __init__(self, rodzic, powrot_do_menu):

        super().__init__(rodzic, padding=10)

        self.powrot_do_menu = powrot_do_menu

        # Stan aplikacji
        self.model = None
        self.slownik = None
        self.id2nazwa = None

        self.obraz_testowy_path = None
        self.trwa_uczenie = False

        # Referencja do obrazu
        self._zdjecie = None

        # Kolejka do komunikacji z wątkiem uczenia
        self._kolejka = queue.Queue()

        # Utworzenie folderów wynikowych
        F_WYNIKI.mkdir(parents=True,exist_ok=True)
        F_ODCZYTANE.mkdir(parents=True,exist_ok=True)

        # Budowanie interfejsu
        self._zbuduj_naglowek()
        self._zbuduj_gorny_panel()
        self._zbuduj_srodkowy_panel()
        self._zbuduj_dolny_panel()

        # Początkowo nie można rozpoznawać obrazów
        self._zablokuj(self.btn_rozpoznaj, True)
        self._zablokuj(self.btn_wybierz, True)
        self._zablokuj(self.btn_zapisz_model, True)

        # Obsługa komunikatów z wątku
        self.after(100,self._obsluz_kolejke)

    # NAGŁÓWEK
    def _zbuduj_naglowek(self):

        naglowek = ttk.Frame(self)
        naglowek.pack(fill="x",pady=(0, 8))

        # Powrót do menu
        self.btn_powrot = ttk.Button(naglowek,text="<- Powrót do menu",command=self.powrot_do_menu)
        self.btn_powrot.pack(side="left",padx=(0, 20))
        ttk.Label(naglowek,text="Klasyfikator dokumentów",font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Button(naglowek,text="Informacje o projekcie",command=self._informacje).pack(side="right")

    def _informacje(self):

        messagebox.showinfo(
            "O projekcie",
            "System klasyfikacji dokumentów "
            "i notatek głosowych.\n\n"
            "Autorzy:\n"
            "Tomasz Chojnacki\n"
            "Maciej Bernatek\n"
            "Przemysław Pałka"
        )

    # PANEL UCZENIA
    def _zbuduj_gorny_panel(self):

        ramka = ttk.LabelFrame(self,text="Uczenie i ewaluacja",padding=8)
        ramka.pack(fill="x",pady=6)

        # Uczenie modelu
        self.btn_ucz = ttk.Button(ramka,text="Rozpocznij proces uczenia",command=self._akcja_uczenie)
        self.btn_ucz.grid(row=0,column=0,padx=5,pady=4)

        # Parametr k
        ttk.Label(ramka,text="k (BoVW):").grid(row=0,column=1,padx=(10, 3))
        self.var_k = tk.IntVar(value=200)
        ttk.Entry(ramka,textvariable=self.var_k,width=7).grid(row=0,column=2)

        # Parametr C
        ttk.Label(ramka,text="C (SVM):").grid(row=0,column=3,padx=(10, 3))
        self.var_C = tk.DoubleVar(value=2.0)
        ttk.Entry(ramka,textvariable=self.var_C,width=7).grid(row=0,column=4)

        # Wczytywanie modelu
        self.btn_wczytaj_model = ttk.Button(ramka,text="Wczytaj zapisany model",command=self._akcja_wczytaj_model)
        self.btn_wczytaj_model.grid(row=0,column=5,padx=12)

        # Pasek postępu
        self.prog = ttk.Progressbar(ramka,mode="indeterminate",length=140)
        self.prog.grid(row=0,column=6,padx=5)

        # Status
        self.lbl_status = ttk.Label(ramka,text="Status: gotowy")
        self.lbl_status.grid(row=1,column=0,columnspan=7,sticky="w",pady=5)

    # PANEL OBRAZU I IDENTYFIKACJI
    def _zbuduj_srodkowy_panel(self):

        kontener = ttk.Frame(self)
        kontener.pack(fill="both",expand=True,pady=4)

        # Dwie równe kolumny
        kontener.columnconfigure(0,weight=1,uniform="polowy")
        kontener.columnconfigure(1,weight=1,uniform="polowy")
        kontener.rowconfigure(0, weight=1)

        # Lewy panel
        lewa = ttk.LabelFrame(kontener,text="Obraz do rozpoznania",padding=8)
        lewa.grid(row=0,column=0,sticky="nsew",padx=(0, 5))
        self.btn_wybierz = ttk.Button(lewa,text="Wybierz obraz…",command=self._akcja_wybierz_obraz)
        self.btn_wybierz.pack(anchor="w",pady=(0, 8))
        self.lbl_podglad = ttk.Label(lewa,text="Nie wybrano obrazu",anchor="center")
        self.lbl_podglad.pack(fill="both",expand=True)

        # Prawy panel
        prawa = ttk.LabelFrame(kontener,text="Identyfikacja",padding=8)
        prawa.grid(row=0,column=1,sticky="nsew",padx=(5, 0))
        self.btn_rozpoznaj = ttk.Button(prawa,text="Wyświetl identyfikację",command=self._akcja_rozpoznaj)
        self.btn_rozpoznaj.pack(anchor="w",pady=(0, 8))
        self.lbl_wynik = ttk.Label(prawa,text="Brak wyniku",font=("Segoe UI", 12, "bold"))
        self.lbl_wynik.pack(anchor="w",pady=8)
        ttk.Separator(prawa).pack(fill="x",pady=8)

        # Zapisywanie modelu
        self.btn_zapisz_model = ttk.Button(prawa,text="Zapisz aktualny model",command=self._akcja_zapisz_model)
        self.btn_zapisz_model.pack(anchor="w",pady=5)

        # Czyszczenie obrazu
        ttk.Button(prawa,text="Wyczyść podgląd",command=self._akcja_wyczysc_podglad).pack(anchor="w",pady=5)

    # PANEL LOGÓW
    def _zbuduj_dolny_panel(self):

        ramka = ttk.LabelFrame(self,text="Log",padding=8)
        ramka.pack(fill="both",pady=(4, 0))
        self.txt_log = tk.Text(ramka,height=9,wrap="word")
        self.txt_log.pack(fill="both",expand=True)
        ttk.Button(ramka,text="Wyczyść log",command=lambda: self.txt_log.delete("1.0","end")).pack(anchor="e",pady=(5, 0))

    # FUNKCJE POMOCNICZE
    @staticmethod
    def _zablokuj(przycisk, blokuj):
        przycisk.configure(state="disabled" if blokuj else "normal")

    def _log(self, tekst):
        self.txt_log.insert("end",tekst + "\n")
        self.txt_log.see("end")

    def _ustaw_status(self, tekst):
        self.lbl_status.configure(text=tekst)
        self._log(tekst)

    def _ustaw_trwa_uczenie(self, trwa):
        self.trwa_uczenie = trwa
        self._zablokuj(self.btn_ucz,trwa)
        self._zablokuj(self.btn_wczytaj_model,trwa)
        self._zablokuj(self.btn_zapisz_model,trwa or self.model is None)
        self._zablokuj(self.btn_wybierz,trwa or self.model is None)
        self._zablokuj(self.btn_rozpoznaj,trwa or self.model is None or self.obraz_testowy_path is None)

        if trwa:
            self.prog.start(10)
        else:
            self.prog.stop()

    # UCZENIE MODELU
    def _akcja_uczenie(self):

        if self.trwa_uczenie:
            return

        # Sprawdzenie folderów
        if not F_TRAIN.is_dir() or not F_TEST.is_dir():
            messagebox.showerror("Błąd",f"Brakuje katalogów:\n"f"{F_TRAIN}\n"f"{F_TEST}")
            return

        # Walidacja parametrów
        try:

            k = self.var_k.get()
            C = self.var_C.get()

            if k <= 0 or C <= 0:
                raise ValueError(
                    "Parametry muszą być dodatnie"
                )

        except (tk.TclError, ValueError):
            messagebox.showerror("Błąd","Parametry k i C muszą być ""dodatnimi liczbami.")

            return

        self._ustaw_trwa_uczenie(True)
        self._ustaw_status("Uczenie w toku…")

        # Osobny wątek uczenia
        def worker():

            try:
                model, slownik, mapa = ucz_model_bovw_svm(F_TRAIN,k=k,C=C)
                self._kolejka.put(("status","Uczenie zakończone. ""Ocena na zbiorze testowym…"))
                acc, f1, _ = ocen_klasyfikator(model,slownik,F_TEST,mapa,F_WYNIKI / "macierz_pomylek.png")
                self._kolejka.put(("sukces",(model,slownik,mapa,acc,f1)))

            except Exception as exc:
                self._kolejka.put(("blad",str(exc)))

        threading.Thread(target=worker,daemon=True).start()

    # OBSŁUGA KOLEJKI
    def _obsluz_kolejke(self):

        try:

            while True:

                rodzaj, dane = self._kolejka.get_nowait()

                if rodzaj == "status":

                    self._ustaw_status(dane)

                elif rodzaj == "sukces":
                    (self.model,self.slownik,self.id2nazwa,acc,f1) = dane
                    self._ustaw_trwa_uczenie(False)
                    self._ustaw_status(f"Accuracy: {acc:.3f} | "f"F1 (macro): {f1:.3f}")

                elif rodzaj == "blad":

                    self._ustaw_trwa_uczenie(False)
                    self._ustaw_status(f"[BŁĄD] {dane}")
                    messagebox.showerror("Błąd podczas uczenia",dane)

        except queue.Empty:
            pass

        self.after(100,self._obsluz_kolejke)

    # WCZYTYWANIE MODELU
    def _akcja_wczytaj_model(self):

        if self.trwa_uczenie:
            return

        try:
            wynik = wczytaj_modeli_slownik(F_WYNIKI)

            if len(wynik) == 3:
                (self.model,self.slownik,self.id2nazwa) = wynik

            else:
                (
                    self.model,
                    self.slownik
                ) = wynik

                self.id2nazwa = (
                    self._odczytaj_mapowanie_z_dysku()
                )

            self._ustaw_trwa_uczenie(False)
            self._ustaw_status("Wczytano zapisany model.")

        except Exception as exc:

            messagebox.showerror("Błąd",f"Nie udało się wczytać modelu:\n{exc}")

    @staticmethod
    def _odczytaj_mapowanie_z_dysku():

        katalogi = sorted(
            p for p in F_TRAIN.iterdir()
            if p.is_dir()
        )

        return {
            i: folder.name
            for i, folder in enumerate(katalogi)
        }

    # ZAPISYWANIE MODELU
    def _akcja_zapisz_model(self):

        if self.model is None or self.slownik is None:
            messagebox.showwarning("Uwaga","Brak modelu do zapisania.")

            return

        try:

            zapisz_modeli_slownik(self.model,self.slownik,self.id2nazwa,F_WYNIKI)
            self._ustaw_status(f"Zapisano model i słownik do: {F_WYNIKI}")

        except Exception as exc:
            messagebox.showerror("Błąd zapisu",str(exc))

    # WYBÓR OBRAZU
    def _akcja_wybierz_obraz(self):

        if self.model is None or self.slownik is None:
            messagebox.showwarning("Uwaga","Najpierw wytrenuj lub wczytaj model.")

            return

        plik = filedialog.askopenfilename(
            title="Wybierz obraz dokumentu",
            filetypes=[("Obrazy","*.jpg *.jpeg *.png ""*.bmp *.tif *.tiff")]
        )

        if not plik:
            return

        try:
            sciezka = Path(plik)
            self._wczytaj_podglad(sciezka)
            self.obraz_testowy_path = sciezka
            self.lbl_wynik.configure(text="Brak wyniku")
            self._zablokuj(self.btn_rozpoznaj,False)
            self._ustaw_status(f"Wybrany obraz: {sciezka.name}")

        except Exception as exc:
            messagebox.showerror("Błąd obrazu",str(exc))


    # PODGLĄD OBRAZU
    def _wczytaj_podglad(self, sciezka):

        with Image.open(sciezka) as img:
            obraz = img.convert("RGB")
            obraz.thumbnail((440, 360))
            self._zdjecie = ImageTk.PhotoImage(obraz)

        self.lbl_podglad.configure(image=self._zdjecie,text="")

    # CZYSZCZENIE PODGLĄDU
    def _akcja_wyczysc_podglad(self):

        self.obraz_testowy_path = None
        self._zdjecie = None
        self.lbl_podglad.configure(image="",text="Nie wybrano obrazu")
        self.lbl_wynik.configure(text="Brak wyniku")
        self._zablokuj(self.btn_rozpoznaj,True)

    # ROZPOZNAWANIE DOKUMENTU
    def _akcja_rozpoznaj(self):

        if (
            self.model is None
            or self.slownik is None
            or self.obraz_testowy_path is None
        ):
            messagebox.showwarning("Uwaga","Brak modelu lub obrazu.")

            return

        try:

            numer = przewidz_etykiete(self.model,self.slownik,self.obraz_testowy_path)
            nazwa = (self.id2nazwa or {}).get(numer,str(numer))
            self.lbl_wynik.configure(text=f"Rozpoznano jako: {nazwa.upper()}")
            self._ustaw_status(f"Wynik: {nazwa}")

            # OCR dla dowodów i biletów
            if nazwa.lower() in ("dowody", "bilety"):
                self._ustaw_status(f"Wykryto {nazwa} – "f"uruchamianie OCR…")

                if nazwa.lower() == "bilety":
                    tekst = ocr_tekst_z_biletu(self.obraz_testowy_path)

                else:
                    tekst = ocr_tekst_z_obrazu(self.obraz_testowy_path)

                # Sprawdzenie odczytanego tekstu
                if not tekst.strip():
                    messagebox.showinfo("OCR","Nie udało się odczytać tekstu.")

                    return

                # Zapis lub wyświetlenie
                if messagebox.askyesno("OCR","Tekst odczytany. ""Zapisać do pliku TXT?"):
                    self._zapisz_tekst_ocr(tekst)

                else:
                    self._log("\n===== TEKST OCR =====\n"+ tekst+ "\n=====================")

        except Exception as exc:
            messagebox.showerror("Błąd",f"Nie udało się rozpoznać obrazu:\n{exc}")
            self._log(f"[BŁĄD] {exc}")


    # ZAPISYWANIE TEKSTU OCR
    def _zapisz_tekst_ocr(self, tekst):

        nazwa_pliku = (f"odczytany_tekst_"f"{self.obraz_testowy_path.stem}.txt")

        try:

            sciezka = F_ODCZYTANE / nazwa_pliku
            sciezka.write_text(tekst,encoding="utf-8")

        except PermissionError:
            katalog = (Path.home() / "Documents" / "odczytane")
            katalog.mkdir(parents=True,exist_ok=True)
            sciezka = katalog / nazwa_pliku
            sciezka.write_text(tekst,encoding="utf-8")

        self._ustaw_status(f"Zapisano OCR: {sciezka}")
        messagebox.showinfo("OCR",f"Zapisano do pliku:\n{sciezka}")
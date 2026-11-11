import queue
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from rozpoznawanie_mowy import (RozpoznawanieMowy,pobierz_mikrofony)

PROJEKT = Path(__file__).resolve().parent.parent
F_NOTATKI = PROJEKT / "notatki"

# WIDOK NOTATNIKA
class WidokNotatnika(ttk.Frame):

    def __init__(self, rodzic, powrot_do_menu):

        super().__init__(rodzic, padding=15)

        self.powrot_do_menu = powrot_do_menu

        F_NOTATKI.mkdir(parents=True,exist_ok=True)

        self.kolejka_wynikow = queue.Queue()
        self.rozpoznawanie = RozpoznawanieMowy(self.kolejka_wynikow)
        self.mikrofony = []
        self.var_mikrofon = tk.StringVar()

        # Tryby:
        # None - bezczynność
        # "test" - test mikrofonu
        # "nagrywanie" - rozpoznawanie mowy
        self.tryb = None
        self.trwa_nagrywanie = False
        self.zatrzymywanie = False
        self.ostatni_blad = False

        # Ścieżka do aktualnie zapisywanej notatki
        # None oznacza, że notatka nie została jeszcze zapisana
        self.sciezka_aktualnej_notatki = None

        self._zbuduj_interfejs()
        self._ustaw_przyciski()

        self.after(200,self.odswiez_mikrofony)
        self.after(100,self._obsluz_kolejke)

    # INTERFEJS
    def _zbuduj_interfejs(self):

        # Nagłówek
        naglowek = ttk.Frame(self)
        naglowek.pack(fill="x",pady=(0, 12))

        ttk.Button(naglowek,text="<- Powrót do menu",command=self._akcja_powrot).pack(side="left",padx=(0, 20))
        ttk.Label(naglowek,text="Notatnik głosowy",font=("Segoe UI", 18, "bold")).pack(side="left")

        # WYBÓR URZĄDZENIA
        panel_mikrofonu = ttk.LabelFrame(self,text="Mikrofon",padding=12)
        panel_mikrofonu.pack(fill="x",pady=(0, 10))
        wiersz_mikrofonu = ttk.Frame(panel_mikrofonu)
        wiersz_mikrofonu.pack(fill="x")
        self.combo_mikrofon = ttk.Combobox(wiersz_mikrofonu,textvariable=self.var_mikrofon,state="readonly")
        self.combo_mikrofon.pack(side="left",fill="x",expand=True,padx=(0, 10))
        self.btn_odswiez = ttk.Button(wiersz_mikrofonu,text="Odśwież",command=self.odswiez_mikrofony)
        self.btn_odswiez.pack(side="right")
        self.lbl_urzadzenia = ttk.Label(panel_mikrofonu,text="Wyszukiwanie urządzeń...")
        self.lbl_urzadzenia.pack(anchor="w",pady=(8, 10))

        # POZIOM MIKROFONU
        ttk.Label(panel_mikrofonu,text="Poziom sygnału mikrofonu:").pack(anchor="w",pady=(0, 5))
        pasek_ramka = ttk.Frame(panel_mikrofonu)
        pasek_ramka.pack(fill="x")
        self.pasek_glosnosci = ttk.Progressbar(pasek_ramka,orient="horizontal",mode="determinate",maximum=100,value=0)
        self.pasek_glosnosci.pack(side="left",fill="x",expand=True,padx=(0, 10))
        self.lbl_poziom = ttk.Label(pasek_ramka,text="0%",width=5,anchor="e")
        self.lbl_poziom.pack(side="right")
        ttk.Label(panel_mikrofonu,text=("Pasek reaguje podczas testu ""lub nagrywania."),foreground="gray").pack(anchor="w",pady=(6, 0))


        # PRZYCISKI NAGRYWANIA
        panel_nagrywania = ttk.LabelFrame(self,text="Nagrywanie",padding=12)
        panel_nagrywania.pack(fill="x",pady=(0, 10))
        self.lbl_status = ttk.Label(panel_nagrywania,text="Status: gotowy do nagrywania",font=("Segoe UI", 11))
        self.lbl_status.pack(anchor="w",pady=(0, 12))
        przyciski = ttk.Frame(panel_nagrywania)
        przyciski.pack(fill="x")

        self.btn_test = ttk.Button(przyciski,text="Testuj mikrofon",command=self._akcja_test)
        self.btn_test.pack(side="left",padx=(0, 10),ipadx=10,ipady=6)

        self.btn_start = ttk.Button(przyciski,text="Rozpocznij nagrywanie",command=self._akcja_start)
        self.btn_start.pack(side="left",padx=(0, 10),ipadx=10,ipady=6)

        self.btn_stop = ttk.Button(przyciski,text="Zatrzymaj",command=self._akcja_stop)
        self.btn_stop.pack(side="left",ipadx=10,ipady=6)

        # PRZYCISKI CZYSZCZENIA
        self.btn_wyczysc = ttk.Button(przyciski,text="Wyczyść notatkę",command=self._akcja_wyczysc)
        self.btn_wyczysc.pack(side="left",padx=(10, 0),ipadx=10,ipady=6)

        # PRZYCISK ZAPISZ
        self.btn_zapisz = ttk.Button(przyciski,text="Zapisz",command=self._akcja_zapisz)
        self.btn_zapisz.pack(side="left",padx=(10, 0),ipadx=10,ipady=6)

        # PRZYCISK ZAPISZ JAKO
        self.btn_zapisz_jako = ttk.Button(przyciski,text="Zapisz jako",command=self._akcja_zapisz_jako)
        self.btn_zapisz_jako.pack(side="left",padx=(10, 0),ipadx=10,ipady=6)

        # POLE TEKSTOWE
        panel_tekstu = ttk.LabelFrame(self,text="Treść notatki",padding=10)
        panel_tekstu.pack(fill="both",expand=True,pady=(0, 10))
        panel_tekstu.columnconfigure(0,weight=1)
        panel_tekstu.rowconfigure(0,weight=1)
        self.txt_notatka = tk.Text(panel_tekstu,font=("Segoe UI", 12),wrap="word",undo=True)
        self.txt_notatka.grid(row=0,column=0,sticky="nsew")
        scrollbar = ttk.Scrollbar(panel_tekstu,orient="vertical",command=self.txt_notatka.yview)
        scrollbar.grid(row=0,column=1,sticky="ns")
        self.txt_notatka.configure(yscrollcommand=scrollbar.set)

        # Styl tekstu tymczasowego Vosk
        self.txt_notatka.tag_configure("tymczasowy",foreground="gray")


    # WYKRYWANIE MIKROFONÓW
    def odswiez_mikrofony(self):

        if self.trwa_nagrywanie:
            return

        try:

            poprzedni_id = self._wybrane_id()
            self.mikrofony = pobierz_mikrofony()
            nazwy = []
            for mikrofon in self.mikrofony:

                opis = (
                    f"[{mikrofon['id']}] "
                    f"{mikrofon['nazwa']} "
                    f"({mikrofon['api']})"
                )

                nazwy.append(opis)

            self.combo_mikrofon["values"] = nazwy

            if not nazwy:

                self.var_mikrofon.set("")
                self.lbl_urzadzenia.configure(text="Nie znaleziono mikrofonów.")
                self._ustaw_przyciski()
                return

            indeks_wybrany = 0

            if poprzedni_id is not None:

                for indeks, mikrofon in enumerate(self.mikrofony):

                    if mikrofon["id"] == poprzedni_id:
                        indeks_wybrany = indeks
                        break

            self.combo_mikrofon.current(indeks_wybrany)
            self.lbl_urzadzenia.configure(text=(f"Wykryto urządzeń wejściowych: "f"{len(self.mikrofony)}"))
            self._ustaw_przyciski()

        except Exception as exc:

            self.mikrofony = []
            self.combo_mikrofon["values"] = ()
            self.var_mikrofon.set("")
            self._ustaw_przyciski()

            messagebox.showerror("Błąd wykrywania mikrofonów",str(exc))

    def _wybrane_id(self):

        indeks = self.combo_mikrofon.current()

        if 0 <= indeks < len(self.mikrofony):
            return self.mikrofony[indeks]["id"]

        return None

    # STANY PRZYCISKÓW
    def _ustaw_przyciski(self):

        aktywne = self.trwa_nagrywanie

        mozna_startowac = (not aktywne and self._wybrane_id() is not None)

        self.btn_start.configure(
            state=(
                "normal"
                if mozna_startowac
                else "disabled"
            )
        )

        self.btn_test.configure(
            state=(
                "normal"
                if mozna_startowac
                else "disabled"
            )
        )

        self.btn_stop.configure(
            state=(
                "normal"
                if aktywne and not self.zatrzymywanie
                else "disabled"
            )
        )

        self.btn_odswiez.configure(
            state=(
                "disabled" if aktywne
                else "normal"
            )
        )

        self.combo_mikrofon.configure(
            state=(
                "disabled" if aktywne
                else "readonly"
            )
        )

        self.btn_zapisz.configure(
            state=(
                "disabled" if aktywne
                else "normal"
            )
        )

        self.btn_zapisz_jako.configure(
            state=(
                "disabled" if aktywne
                else "normal"
            )
        )

        self.btn_wyczysc.configure(
            state=(
                "disabled" if aktywne
                else "normal"
            )
        )

        # W czasie nagrywania pole jest
        # aktualizowane przez program.
        # Po zatrzymaniu można je edytować.
        self.txt_notatka.configure(
            state=(
                "disabled"
                if self.tryb == "nagrywanie"
                else "normal"
            )
        )

    # ROZPOCZĘCIE PRACY MIKROFONU
    def _uruchom(self, test):

        if self.trwa_nagrywanie:
            return

        mikrofon_id = self._wybrane_id()

        if mikrofon_id is None:

            messagebox.showwarning("Brak mikrofonu","Wybierz mikrofon z listy.")
            return

        self.ostatni_blad = False
        self.zatrzymywanie = False

        # Nie usuwamy poprzedniej notatki.
        # Nowe słowa zostaną dopisane na końcu.
        uruchomiono = self.rozpoznawanie.rozpocznij(mikrofon_id,test=test)

        if not uruchomiono:
            return

        self.tryb = (
            "test" if test
            else "nagrywanie"
        )

        self.trwa_nagrywanie = True
        self.lbl_status.configure(text="Status: uruchamianie mikrofonu...")
        self._ustaw_przyciski()

    def _akcja_test(self):

        self._uruchom(test=True)

    def _akcja_start(self):

        self._uruchom(test=False)

    # ZATRZYMANIE
    def _akcja_stop(self):

        if not self.trwa_nagrywanie:
            return

        self.zatrzymywanie = True
        self.lbl_status.configure(text="Status: zatrzymywanie...")
        self.rozpoznawanie.zatrzymaj()
        self._ustaw_przyciski()

    # OBSŁUGA KOMUNIKATÓW Z WĄTKU
    def _obsluz_kolejke(self):
        try:
            while True:
                rodzaj, dane = (self.kolejka_wynikow.get_nowait())

                if rodzaj == "status":
                    self.lbl_status.configure(text=f"Status: {dane}")

                elif rodzaj == "rozpoczeto":

                    if dane == "test":
                        self.lbl_status.configure(text=("Status: test mikrofonu ""rozpoczęty"))

                    else:
                        self.lbl_status.configure(text=("Status: nagrywanie ""rozpoczęte"))

                elif rodzaj == "poziom":
                    self._aktualizuj_poziom(dane)

                elif rodzaj == "fragment":
                    self._wyswietl_fragment(dane)

                elif rodzaj == "tekst":
                    self._dodaj_tekst(dane)

                elif rodzaj == "ostrzezenie":
                    self.lbl_status.configure(text=f"Uwaga: {dane}")

                elif rodzaj == "blad":
                    self.ostatni_blad = True
                    self.lbl_status.configure(text="Status: błąd mikrofonu")
                    messagebox.showerror("Błąd mikrofonu",dane)

                elif rodzaj == "koniec":
                    self._zakoncz_prace()

        except queue.Empty:
            pass

        self.after(100,self._obsluz_kolejke)

    # WSKAŹNIK GŁOŚNOŚCI
    def _aktualizuj_poziom(self, poziom):

        poziom = max(0,min(100, int(poziom)))
        self.pasek_glosnosci["value"] = poziom
        self.lbl_poziom.configure(text=f"{poziom}%")

    # POMOCNICZA EDYCJA POLA TEKSTOWEGO
    def _edytuj_programowo(self, operacja):

        poprzedni_stan = str(self.txt_notatka.cget("state"))
        self.txt_notatka.configure(state="normal")

        try:
            operacja()
        finally:
            self.txt_notatka.configure(state=poprzedni_stan)

    # USUWANIE TYMCZASOWEGO FRAGMENTU
    def _usun_fragment(self):

        zakres = self.txt_notatka.tag_ranges("tymczasowy")

        if len(zakres) == 2:

            self.txt_notatka.delete(zakres[0],zakres[1])

    # TEKST WYŚWIETLANY NA BIEŻĄCO
    def _wyswietl_fragment(self, fragment):

        if self.tryb != "nagrywanie":
            return

        fragment = fragment.strip()

        def aktualizacja():

            # Najpierw usuwamy poprzedni
            # fragment tymczasowy.
            self._usun_fragment()

            if not fragment:
                return

            aktualny = self.txt_notatka.get("1.0","end-1c")

            separator = (
                " " if aktualny and not aktualny.endswith(
                    (" ", "\n")
                ) else ""
            )

            poczatek = self.txt_notatka.index("end-1c")

            # Wstawiamy aktualną transkrypcję
            # w tym samym polu notatki.
            self.txt_notatka.insert("end",separator + fragment)

            koniec = self.txt_notatka.index("end-1c")

            # Znacznik fragmentu tymczasowego
            self.txt_notatka.tag_add("tymczasowy",poczatek,koniec)
            self.txt_notatka.see("end")

        self._edytuj_programowo(aktualizacja)

    # ZATWIERDZANIE ROZPOZNANEGO TEKSTU
    def _dodaj_tekst(self, tekst):

        if self.tryb != "nagrywanie":
            return

        tekst = tekst.strip()

        if not tekst:
            return

        def aktualizacja():

            # Usunięcie poprzedniej wersji
            # częściowej tego samego fragmentu.
            self._usun_fragment()

            aktualny = self.txt_notatka.get("1.0","end-1c")

            separator = (
                " " if aktualny and not aktualny.endswith(
                    (" ", "\n")
                ) else ""
            )

            self.txt_notatka.insert("end",separator + tekst)
            self.txt_notatka.see("end")
        self._edytuj_programowo(aktualizacja)

    # ZAKOŃCZENIE NAGRYWANIA / TESTU
    def _zakoncz_prace(self):

        poprzedni_tryb = self.tryb

        self.trwa_nagrywanie = False
        self.zatrzymywanie = False
        self.tryb = None

        # Usuwamy ewentualne słowa, które
        # pozostały jedynie wynikiem częściowym.
        self._edytuj_programowo(self._usun_fragment)
        self._aktualizuj_poziom(0)
        self._ustaw_przyciski()

        if self.ostatni_blad:
            return

        if poprzedni_tryb == "test":
            self.lbl_status.configure(text="Status: test mikrofonu zakończony")

        else:
            self.lbl_status.configure(text=("Status: nagrywanie zakończone ""– można edytować lub wznowić"))

    # ZAPIS DO PLIKU
    def _zapisz_do_pliku(self, sciezka):

        # Nie zapisujemy podczas nagrywania
        if self.trwa_nagrywanie:
            return False

        tekst = self.txt_notatka.get("1.0","end-1c")

        if not tekst.strip():
            messagebox.showwarning("Pusta notatka","Nie ma tekstu do zapisania.")

            return False

        try:

            sciezka = Path(sciezka)

            # Zapis aktualnej, pełnej treści notatki.
            # Jeśli plik już istnieje, jego treść
            # zostanie zastąpiona nową wersją.

            sciezka.write_text(tekst,encoding="utf-8")

            # Zapamiętanie ścieżki
            self.sciezka_aktualnej_notatki = sciezka

            # Aktualizacja statusu
            self.lbl_status.configure(text=("Status: zapisano notatkę – "f"{sciezka.name}"))

            return True

        except Exception as exc:
            messagebox.showerror("Błąd zapisu",f"Nie udało się zapisać notatki:\n{exc}")

            return False

    # PRZYCISK ZAPISZ
    def _akcja_zapisz(self):

        if self.trwa_nagrywanie:
            return

        # Jeśli notatka nie ma jeszcze pliku,
        # uruchamiamy funkcję "Zapisz jako".

        if self.sciezka_aktualnej_notatki is None:
            self._akcja_zapisz_jako()
            return

        # Jeśli plik już istnieje,
        # zapisujemy aktualną treść do tego pliku.

        self._zapisz_do_pliku(self.sciezka_aktualnej_notatki)

    # PRZYCISK ZAPISZ JAKO
    def _akcja_zapisz_jako(self):

        if self.trwa_nagrywanie:
            return

        tekst = self.txt_notatka.get("1.0","end-1c")

        if not tekst.strip():
            messagebox.showwarning("Pusta notatka","Nie ma tekstu do zapisania.")

            return

        # Katalog początkowy
        katalog = F_NOTATKI

        # Domyślna nazwa
        nazwa_pliku = "notatka.txt"

        # Jeśli wcześniej zapisano notatkę,
        # pokazujemy jej nazwę i katalog.

        if self.sciezka_aktualnej_notatki is not None:

            katalog = (self.sciezka_aktualnej_notatki.parent)

            nazwa_pliku = (self.sciezka_aktualnej_notatki.name)

        # Okno wyboru lokalizacji
        sciezka = filedialog.asksaveasfilename(
            title="Zapisz notatkę jako",
            initialdir=str(katalog),
            initialfile=nazwa_pliku,
            defaultextension=".txt",
            filetypes=[
                ("Pliki tekstowe", "*.txt")
            ]
        )

        if not sciezka:
            return

        # Zapisanie wybranej notatki
        if self._zapisz_do_pliku(sciezka):
            messagebox.showinfo("Zapisano notatkę","Notatka została zapisana:\n"f"{sciezka}")


    # CZYSZCZENIE NOTATKI
    def _akcja_wyczysc(self):

        if self.trwa_nagrywanie:
            return

        tekst = self.txt_notatka.get("1.0","end-1c").strip()

        # Wyczyszczona notatka będzie traktowana
        # jako nowa notatka, bez przypisanego pliku.

        self.sciezka_aktualnej_notatki = None

        if tekst:
            decyzja = messagebox.askyesno("Wyczyść notatkę","Czy usunąć całą treść notatki?")

            if not decyzja:
                return

        self.txt_notatka.delete("1.0","end")
        self.lbl_status.configure(text="Status: notatka wyczyszczona")

    # POWRÓT DO MENU
    def _akcja_powrot(self):

        if self.trwa_nagrywanie:

            self.rozpoznawanie.zatrzymaj()
            self.zatrzymywanie = True
            self._ustaw_przyciski()

        # Notatka pozostaje w pamięci widoku.
        # Nie zapisujemy jej automatycznie.
        self.powrot_do_menu()

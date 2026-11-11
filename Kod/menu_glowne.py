from tkinter import ttk

class MenuGlowne(ttk.Frame):

    def __init__(self,rodzic,otworz_klasyfikator,otworz_notatnik):

        super().__init__(rodzic, padding=24)

        # Dwie kolumny o jednakowej szerokości
        self.columnconfigure(0,weight=1,uniform="moduly")
        self.columnconfigure(1,weight=1,uniform="moduly")

        self.rowconfigure(1, weight=1)

        # Nagłówek aplikacji
        naglowek = ttk.Frame(self)
        naglowek.grid(row=0,column=0,columnspan=2,pady=(12, 30))
        ttk.Label(naglowek,text="System klasyfikacji dokumentów " "i notatek głosowych",font=("Segoe UI", 19, "bold")).pack(pady=(0, 8))
        ttk.Label(naglowek,text="Wybierz moduł, ""z którego chcesz skorzystać",font=("Segoe UI", 12)).pack()

        # Lewy panel - klasyfikator
        self._utworz_panel(
            0,
            "DOKUMENTY",
            "Klasyfikator dokumentów",
            "Rozpoznawanie rodzaju dokumentów "
            "za pomocą ORB, BoVW i SVM.\n"
            "Odczytywanie tekstu "
            "przy użyciu Tesseract OCR.",
            "Otwórz klasyfikator",
            otworz_klasyfikator
        )

        # Prawy panel - notatnik
        self._utworz_panel(
            1,
            "NOTATKI",
            "Notatnik głosowy",
            "Nagrywanie wypowiedzi z mikrofonu.\n"
            "Zamiana mowy na tekst i ręczne "
            "zapisywanie notatek.",
            "Otwórz notatnik",
            otworz_notatnik
        )

    def _utworz_panel(self,kolumna,ikona,tytul,opis,przycisk,akcja):

        panel = ttk.LabelFrame(self,text=ikona,padding=28)
        panel.grid(row=1,column=kolumna,sticky="nsew",padx=10,pady=10)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        # Środkowy obszar panelu
        srodek = ttk.Frame(panel)
        srodek.grid(row=0, column=0)

        # Tytuł
        ttk.Label(srodek,text=tytul,font=("Segoe UI", 17, "bold"),anchor="center").pack(pady=(0, 22))

        # Opis
        ttk.Label(srodek,text=opis,font=("Segoe UI", 11),anchor="center",justify="center",wraplength=380).pack(pady=(0, 28))

        # Przycisk przejścia do modułu
        ttk.Button(srodek,text=przycisk,command=akcja).pack(ipadx=25,ipady=10)
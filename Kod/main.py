import tkinter as tk
from tkinter import ttk

from menu_glowne import MenuGlowne
from ui_klasyfikator import WidokKlasyfikatora
from ui_notatnik import WidokNotatnika

class Aplikacja(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title("System klasyfikacji dokumentów ""i notatek głosowych")

        self.geometry("1100x740")
        self.minsize(900, 620)

        # GŁÓWNY KONTENER
        self.kontener = ttk.Frame(self)
        self.kontener.pack(fill="both",expand=True)
        self.kontener.grid_rowconfigure(0,weight=1)
        self.kontener.grid_columnconfigure(0,weight=1)

        # TWORZENIE WIDOKÓW
        # Menu główne
        self.menu_glowne = MenuGlowne(self.kontener,self.pokaz_klasyfikator,self.pokaz_notatnik)

        # Klasyfikator dokumentów
        self.klasyfikator = WidokKlasyfikatora(self.kontener,self.pokaz_menu)

        # Notatnik głosowy
        self.notatnik = WidokNotatnika(self.kontener,self.pokaz_menu)

        # ROZMIESZCZENIE WIDOKÓW
        for widok in (self.menu_glowne,self.klasyfikator,self.notatnik):
            widok.grid(row=0,column=0,sticky="nsew")

        # Ekran startowy
        self.pokaz_menu()

        # Bezpieczne zamknięcie aplikacji
        self.protocol("WM_DELETE_WINDOW",self._zamknij_aplikacje)

    # MENU GŁÓWNE
    def pokaz_menu(self):
        self.menu_glowne.tkraise()

    # KLASYFIKATOR
    def pokaz_klasyfikator(self):
        self.klasyfikator.tkraise()

    # NOTATNIK GŁOSOWY
    def pokaz_notatnik(self):
        # Wyświetlenie notatnika
        self.notatnik.tkraise()

        # Odświeżenie listy mikrofonów
        if not self.notatnik.trwa_nagrywanie:
            self.notatnik.odswiez_mikrofony()

    # ZAMKNIĘCIE APLIKACJI
    def _zamknij_aplikacje(self):

        # Zatrzymanie mikrofonu
        if self.notatnik.trwa_nagrywanie:
            self.notatnik.rozpoznawanie.zatrzymaj()

        self.destroy()

if __name__ == "__main__":

    aplikacja = Aplikacja()
    aplikacja.mainloop()
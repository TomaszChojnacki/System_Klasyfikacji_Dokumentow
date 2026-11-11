import json
import math
import queue
import threading
from pathlib import Path

import numpy as np

PROJEKT = Path(__file__).resolve().parent.parent

SCIEZKA_MODELU = (PROJEKT/ "modele_vosk"/ "vosk-model-small-pl-0.22")


# WYKRYWANIE MIKROFONÓW
def pobierz_mikrofony():
    import sounddevice as sd

    mikrofony = []
    urzadzenia = sd.query_devices()
    hostapis = sd.query_hostapis()

    for indeks, urzadzenie in enumerate(urzadzenia):

        if urzadzenie["max_input_channels"] < 1:
            continue

        nazwa_api = hostapis[urzadzenie["hostapi"]]["name"]
        mikrofony.append({"id": indeks,"nazwa": urzadzenie["name"],"api": nazwa_api,"samplerate": int(urzadzenie["default_samplerate"])})

    return mikrofony


# ROZPOZNAWANIE MOWY
class RozpoznawanieMowy:

    def __init__(self, kolejka_wynikow):

        self.kolejka_wynikow = kolejka_wynikow

        self.kolejka_audio = queue.Queue()

        self.stop_event = threading.Event()
        self.blokada = threading.Lock()

        self.watek = None
        self.model = None

        self.czy_nagrywa = False
        self.tryb_testu = False

        self.licznik_poziomu = 0

    # URUCHOMIENIE
    def rozpocznij(self, id_mikrofonu, test=False):
        """
        test=False: rozpoznawanie mowy
        test=True: tylko test mikrofonu
        """

        with self.blokada:

            if self.czy_nagrywa:
                return False

            self.czy_nagrywa = True
            self.tryb_testu = test

            self.stop_event.clear()
            self.kolejka_audio = queue.Queue()
            self.licznik_poziomu = 0

            self.watek = threading.Thread(target=self._pracuj,args=(int(id_mikrofonu), test),daemon=True)

            self.watek.start()

        return True

    # ZATRZYMANIE
    def zatrzymaj(self):

        self.stop_event.set()


    # POMIAR GŁOŚNOŚCI
    @staticmethod
    def oblicz_poziom(dane):

        probki = np.frombuffer(dane,dtype=np.int16)

        if len(probki) == 0:
            return 0

        probki = probki.astype(np.float64)
        rms = np.sqrt(np.mean(probki ** 2))

        if rms <= 0:
            return 0

        # Poziom sygnału w dBFS
        db = 20 * math.log10(max(rms / 32768.0, 1e-10))

        # Zakres wizualizacji: -55 do -5 dBFS
        poziom = (db + 55) * 2

        return int(max(0, min(100, poziom)))

    # CALLBACK MIKROFONU
    def _callback_audio(self,indata,frames,time_info,status):

        if self.stop_event.is_set():
            return

        dane = bytes(indata)

        if status:
            self.kolejka_wynikow.put(("ostrzezenie", str(status)))

        # Pomiar poziomu dźwięku
        self.licznik_poziomu += 1

        # Nie wysyłamy każdej próbki do UI,
        # żeby nie przeciążać Tkinter.
        if self.licznik_poziomu >= 3:

            self.licznik_poziomu = 0
            poziom = self.oblicz_poziom(dane)
            self.kolejka_wynikow.put(("poziom", poziom))

        # W trybie testowym nie wykonujemy OCR mowy
        if not self.tryb_testu:

            self.kolejka_audio.put(dane)


    # OBSŁUGA MIKROFONU I VOSK
    def _pracuj(self, id_mikrofonu, test):

        try:
            import sounddevice as sd

            urzadzenie = sd.query_devices(id_mikrofonu,kind="input")
            czestotliwosc = int(urzadzenie["default_samplerate"])

            if czestotliwosc <= 0:
                raise RuntimeError("Nieprawidłowa częstotliwość ""próbkowania mikrofonu.")

            sd.check_input_settings(device=id_mikrofonu,channels=1,dtype="int16",samplerate=czestotliwosc)

            rozpoznawacz = None

            if not test:
                from vosk import (Model,KaldiRecognizer,SetLogLevel)

                SetLogLevel(-1)

                if not SCIEZKA_MODELU.is_dir():
                    raise FileNotFoundError("Nie znaleziono modelu Vosk:\n"f"{SCIEZKA_MODELU}")

                self.kolejka_wynikow.put(("status","Ładowanie modelu Vosk..."))

                # Wczytanie modelu tylko raz
                if self.model is None:
                    self.model = Model(
                        str(SCIEZKA_MODELU)
                    )

                rozpoznawacz = KaldiRecognizer(self.model,czestotliwosc)

            if self.stop_event.is_set():
                return

            # Uruchomienie mikrofonu
            with sd.RawInputStream(device=id_mikrofonu,samplerate=czestotliwosc,blocksize=2048,dtype="int16",channels=1,callback=self._callback_audio):

                self.kolejka_wynikow.put(("rozpoczeto","test" if test else "nagrywanie"))

                # TRYB TESTOWANIA MIKROFONU
                if test:

                    while not self.stop_event.wait(0.1):
                        pass

                # TRYB ROZPOZNAWANIA MOWY
                else:

                    ostatni_fragment = ""
                    # Przetwarzamy również próbki,
                    # które pozostały w kolejce
                    # w chwili zatrzymania.
                    while (not self.stop_event.is_set()or not self.kolejka_audio.empty()):

                        try:
                            dane = self.kolejka_audio.get(
                                timeout=0.1
                            )

                        except queue.Empty:
                            continue

                        if rozpoznawacz.AcceptWaveform(dane):
                            wynik = json.loads(rozpoznawacz.Result())

                            tekst = wynik.get("text", "").strip()

                            if tekst:
                                self.kolejka_wynikow.put(("tekst", tekst))

                            ostatni_fragment = ""
                            self.kolejka_wynikow.put(("fragment", ""))

                        else:
                            wynik = json.loads(rozpoznawacz.PartialResult())

                            fragment = wynik.get("partial", "").strip()

                            if fragment != ostatni_fragment:
                                ostatni_fragment = fragment
                                self.kolejka_wynikow.put(("fragment", fragment))

            # OSTATNI FRAGMENT TRANSKRYPCJI
            if not test and rozpoznawacz is not None:
                wynik = json.loads(rozpoznawacz.FinalResult())
                tekst = wynik.get("text", "").strip()

                if tekst:
                    self.kolejka_wynikow.put(("tekst", tekst))

                self.kolejka_wynikow.put(("fragment", ""))

        except Exception as exc:
            self.kolejka_wynikow.put(("blad", str(exc)))

        finally:
            with self.blokada:
                self.czy_nagrywa = False

            self.kolejka_wynikow.put(("poziom", 0))
            self.kolejka_wynikow.put(("koniec", None))

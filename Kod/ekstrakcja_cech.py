from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from PIL import Image

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# Funkcja konwertuje obraz z dysku dzięki temu obraz ma ten sam format wejściowy.
def wczytaj_obraz_szaroscii(sciezka, rozmiar=(1920,1080)):
    img = Image.open(sciezka).convert("RGB")  # usuwa błędny profil ICC
    img = img.resize(rozmiar)
    return np.array(img.convert("L"))         #do skali szarości L=luminancja


# Funkcja wykrywa punkty kluczowe przez ORB Zwraca zestaw wektorów opisujących lokalne fragmenty obrazu.
def wykryj_opisy_orb(obraz_szary: np.ndarray, max_kluczowych: int = 1000) -> Optional[np.ndarray]:
    orb = cv2.ORB_create(nfeatures=max_kluczowych)   # inicjalizacja ORB
    kluczowe, opisy = orb.detectAndCompute(obraz_szary, None)  # wykrycie punktów i opisów
    return opisy


# Funkcja buduje słownik BoVW zbiór słów wizualnych
# Każde słowo przedstawia typowy lokalny wzorzec obrazu
# Używa do tego algorytmu KMeans, który grupuje cechy ORB.
def zbuduj_slownik_bovw(lista_sciezek: List[Path], k: int = 200, rozmiar_wejscia: Tuple[int,int]=(800,800)) -> KMeans:
    wszystkie_opisy = []   # lista wszystkich opisów z wszystkich obrazów

    for p in lista_sciezek:
        if p.suffix.lower() not in ROZSZERZENIA:
            continue
        img = wczytaj_obraz_szaroscii(p, rozmiar_wejscia)
        if img is None:
            continue
        opisy = wykryj_opisy_orb(img)
        if opisy is not None:
            wszystkie_opisy.append(opisy.astype(np.float32))  # zapisanie cech

    # Nie znaleziono żadnych cech = zatrzymaj program
    if not wszystkie_opisy:
        raise RuntimeError("Nie zebrano żadnych opisów ORB — sprawdź dane wejściowe.")

    # Łączymy opisy w jedną macierz (każdy wiersz to opis cechy)
    macierz_opisow = np.vstack(wszystkie_opisy)

    # Tworzymy model KMeans – grupuje cechy w k klastrów (słów wizualnych)
    kmeans = KMeans(n_clusters=k, n_init=8, max_iter=300, verbose=0, random_state=0)
    kmeans.fit(macierz_opisow)
    return kmeans


# Funkcja tworzy histogram BoVW który mówi, jak często w obrazie występują poszczególne słowa wizualne.
# Jest to końcowa reprezentacja obrazu używana przez klasyfikator SVM.
def histogram_bovw(opisy: np.ndarray, slownik: KMeans) -> np.ndarray:
    # Jeśli obraz nie ma cech (np. słabe zdjęcie), zwraca pusty histogram
    if opisy is None or len(opisy) == 0:
        hist = np.zeros(slownik.n_clusters, dtype=np.float32)
        return hist

    # Przypisujemy każdy opis do najbliższego „słowa” w słowniku
    indeksy = slownik.predict(opisy.astype(np.float32))

    # Liczymy, ile razy każde słowo wystąpiło (histogram częstości)
    hist, _ = np.histogram(indeksy, bins=np.arange(slownik.n_clusters + 1))

    # Normalizujemy histogram, żeby suma długości była 1 (niezależnie od liczby cech)
    hist = hist.astype(np.float32)[None, :]
    hist = normalize(hist, norm="l2")
    return hist.ravel()  # zwracamy spłaszczony wektor (1D)

"""
Do usuniecia kom po zaakceptowaniu prototypu
Ekstrakcja cech obrazu i budowa słownika wizualnego (BoVW) na bazie ORB.
- wykryj_opisy_orb: zwraca opisy ORB dla jednego obrazu
- zbuduj_slownik_bovw: uczy KMeans i zwraca obiekt k-średnich (słownik wizualny)
- histogram_bovw: zamienia opisy ORB na histogram "słów wizualnych"
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from PIL import Image

# Dozwolone rozszerzenia obrazów
ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def wczytaj_obraz_szaroscii(sciezka, rozmiar=(1920,1080)):
    img = Image.open(sciezka).convert("RGB")  # usuwa błędny profil ICC
    img = img.resize(rozmiar)
    return np.array(img.convert("L"))

def wykryj_opisy_orb(obraz_szary: np.ndarray, max_kluczowych: int = 1000) -> Optional[np.ndarray]:
    """
    Zwraca tablicę opisów ORB (N x 32) albo None, jeśli nie znaleziono punktów.
    """
    orb = cv2.ORB_create(nfeatures=max_kluczowych)
    kluczowe, opisy = orb.detectAndCompute(obraz_szary, None)
    return opisy

def zbuduj_slownik_bovw(lista_sciezek: List[Path], k: int = 200, rozmiar_wejscia: Tuple[int,int]=(800,800)) -> KMeans:
    """
    Buduje słownik wizualny (KMeans) na podstawie opisów ORB z wielu obrazów.
    """
    wszystkie_opisy = []
    for p in lista_sciezek:
        if p.suffix.lower() not in ROZSZERZENIA:
            continue
        img = wczytaj_obraz_szaroscii(p, rozmiar_wejscia)
        if img is None:
            continue
        opisy = wykryj_opisy_orb(img)
        if opisy is not None:
            wszystkie_opisy.append(opisy.astype(np.float32))

    if not wszystkie_opisy:
        raise RuntimeError("Nie zebrano żadnych opisów ORB — sprawdź dane wejściowe.")

    macierz_opisow = np.vstack(wszystkie_opisy)  # (M x 32)
    # KMeans: losowe inicjalizacje i trochę iteracji w zupełności wystarczy
    kmeans = KMeans(n_clusters=k, n_init=8, max_iter=300, verbose=0, random_state=0)
    kmeans.fit(macierz_opisow)
    return kmeans

def histogram_bovw(opisy: np.ndarray, slownik: KMeans) -> np.ndarray:
    """
    Do usuniecia kom po zaakceptowaniu prototypu
    Zamienia opisy (N x 32) na histogram słów wizualnych (1 x k) i normalizuje L2.
    """
    if opisy is None or len(opisy) == 0:
        hist = np.zeros(slownik.n_clusters, dtype=np.float32)
        return hist
    indeksy = slownik.predict(opisy.astype(np.float32))  # do którego „słowa” należy każdy opis
    hist, _ = np.histogram(indeksy, bins=np.arange(slownik.n_clusters + 1))
    hist = hist.astype(np.float32)[None, :]  # (1 x k)
    hist = normalize(hist, norm="l2")        # normalizacja
    return hist.ravel()


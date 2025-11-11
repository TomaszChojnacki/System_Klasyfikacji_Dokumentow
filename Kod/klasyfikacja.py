from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from ekstrakcja_cech import (
    ROZSZERZENIA,
    wczytaj_obraz_szaroscii,
    wykryj_opisy_orb,
    zbuduj_slownik_bovw,
    histogram_bovw
)

# Funkcja wczytuje dane ze struktury folderów (np. dane/trening/faktury, paragony itd.)
# Zwraca listę ścieżek do plików, listę etykiet numerycznych i mapę id→nazwa klasy
def wczytaj_zbior_ze_struktury(folder_glowny: Path) -> Tuple[List[Path], List[int], Dict[int, str]]:
    sciezki: List[Path] = []
    etykiety: List[int] = []
    mapa_id2nazwa: Dict[int, str] = {}

    podfoldery = sorted([p for p in folder_glowny.iterdir() if p.is_dir()])
    for idx, klasa_dir in enumerate(podfoldery):
        mapa_id2nazwa[idx] = klasa_dir.name
        for plik in klasa_dir.iterdir():
            if plik.suffix.lower() in ROZSZERZENIA:
                sciezki.append(plik)
                etykiety.append(idx)
    return sciezki, etykiety, mapa_id2nazwa


# Funkcja uczy model klasyfikatora dokumentów:
# 1. Tworzy słownik BoVW (słowa wizualne)
# 2. Ekstrahuje cechy ORB dla wszystkich obrazów
# 3. Trenuje klasyfikator SVM
def ucz_model_bovw_svm(
    folder_trening: Path,
    k: int = 200,
    C: float = 1.0,
    gamma: str | float = "scale",
) -> Tuple[SVC, object, Dict[int, str]]:
    sciezki, y, id2nazwa = wczytaj_zbior_ze_struktury(folder_trening)
    slownik = zbuduj_slownik_bovw(sciezki, k=k)

    X_hist = []
    for p in sciezki:
        img = wczytaj_obraz_szaroscii(p, (800, 800))
        opisy = wykryj_opisy_orb(img)
        X_hist.append(histogram_bovw(opisy, slownik))
    X = np.vstack(X_hist)

    clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=False, random_state=0)
    clf.fit(X, y)
    return clf, slownik, id2nazwa


# Funkcja przewiduje etykietę klasy dla nowego obrazu
# (czyli rozpoznaje, jaki to typ dokumentu)
def przewidz_etykiete(model: SVC, slownik, sciezka: Path) -> int:
    img = wczytaj_obraz_szaroscii(sciezka, (800, 800))
    opisy = wykryj_opisy_orb(img)
    hist = histogram_bovw(opisy, slownik).reshape(1, -1)
    etyk = model.predict(hist)[0]
    return int(etyk)


# Funkcja ocenia skuteczność klasyfikatora:
# oblicza Accuracy, F1-score i tworzy macierz pomyłek zapisaną do pliku .png
def ocen_klasyfikator(
    model: SVC,
    slownik,
    folder_test: Path,
    id2nazwa: Dict[int, str],
    sciezka_obrazu_cm: Path = Path("wyniki/macierz_pomylek.png")
    ) -> Tuple[float, float, np.ndarray]:
    sciezki, y_true, _ = wczytaj_zbior_ze_struktury(folder_test)
    X_test = []
    for p in sciezki:
        img = wczytaj_obraz_szaroscii(p, (800, 800))
        opisy = wykryj_opisy_orb(img)
        X_test.append(histogram_bovw(opisy, slownik))
    X_test = np.vstack(X_test)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    # Rysowanie i zapis macierzy pomyłek
    etykiety_opisowe = [id2nazwa[i] for i in sorted(id2nazwa.keys())]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Macierz pomyłek")
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Rzeczywista")
    ax.set_xticks(range(len(etykiety_opisowe)))
    ax.set_yticks(range(len(etykiety_opisowe)))
    ax.set_xticklabels(etykiety_opisowe, rotation=45, ha="right")
    ax.set_yticklabels(etykiety_opisowe)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    sciezka_obrazu_cm.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sciezka_obrazu_cm)
    plt.close(fig)

    return acc, f1, cm


# Funkcja zapisuje wytrenowany model, słownik BoVW i mapę klas do plików .joblib
def zapisz_modeli_slownik(model: SVC, slownik, id2nazwa: Dict[int, str], katalog: Path = Path("wyniki")) -> None:
    katalog.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, katalog / "model_svm.joblib")
    joblib.dump(slownik, katalog / "slownik_kmeans.joblib")
    joblib.dump(id2nazwa, katalog / "mapa_id2nazwa.joblib")


# Funkcja wczytuje wcześniej zapisany model, słownik i mapę klas
# (pozwala kontynuować działanie programu bez ponownego uczenia)
def wczytaj_modeli_slownik(katalog: Path = Path("wyniki")) -> tuple:
    model = joblib.load(katalog / "model_svm.joblib")
    slownik = joblib.load(katalog / "slownik_kmeans.joblib")
    id2nazwa = joblib.load(katalog / "mapa_id2nazwa.joblib")
    return model, slownik, id2nazwa

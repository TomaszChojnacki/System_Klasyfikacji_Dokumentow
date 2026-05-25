# ocr.py
from pathlib import Path
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"D:\Projekt\System_Klasyfikacji_Dokumentow\Tesseract\tesseract.exe"

def ocr_tekst_z_obrazu(sciezka: Path) -> str:
    img = Image.open(sciezka)
    img = img.convert("L")  # konwersja na skalę szarości

    # Konfiguracja OCR
    config = (
        "--oem 3 "  # domyślny silnik
        "--psm 11 " # traktuj obraz jako pojedynczą linię
    )

    tekst = pytesseract.image_to_string(img, lang="pol", config=config)
    return tekst.strip()

def ocr_tekst_z_biletu(sciezka: Path) -> str:
    img = Image.open(sciezka).convert("L")

    # powiększenie obrazu poprawia OCR
    img = img.resize((img.width * 2, img.height * 2))

    # OCR próbujemy dla kilku obrotów, bo bilety często są bokiem
    najlepszy_tekst = ""

    for kat in [0, 90, 180, 270]:
        obrocony = img.rotate(kat, expand=True)

        config = (
            "--oem 3 " # domyślny silnik
            "--psm 11 "
        )

        tekst = pytesseract.image_to_string(obrocony, lang="pol", config=config).strip()

        if len(tekst) > len(najlepszy_tekst):
            najlepszy_tekst = tekst

    return najlepszy_tekst
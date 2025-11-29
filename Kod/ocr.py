# ocr.py
from pathlib import Path
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Maciek\Desktop\studia\sys rozpoznawania mowy i obrazu\System_Klasyfikacji_Dokumentow\Tesseract\tesseract.exe"

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

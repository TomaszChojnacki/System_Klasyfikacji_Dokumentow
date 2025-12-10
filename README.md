
**SYSTEM KLASYFIKACJI DOKUMENTOW**
# I. Opis projektu 
Glownym celem projektu bylo stworzyc program do rozpoznawania roznego rodzaju dokumentow na podstawie obrazu oraz wyciaganie z nich tekstu. Zalozylismy odgornie, ze bedziemy rozpoznawac piec rodzajow dokumentow w naszym projekcie - dokumenty osobiste, faktury, wizytowki, paragony, CV. Nastepnie program po klasyfikacji zdjecia do jednego z 5 kategorii mial odczytywac tekst z obrazu. Ze wzgledu na zlozonosc i roznorodnosc tych rodzajow dokumentow stwierdzilismy, ze bedziemy odczytywac informacje tylko z dowod osobistych.

-----

## II.  Instrukcja instalacji bibliotek

pip install opencv-python
pip install scikit-learn
pip install numpy
pip install pillow
pip install matplotlib
pip install pytesseract
pip install joblib

jesli dalej brakuje bibliotek to (jesli uzywamy pycharma) to at+enter, czyli show context actions mozemy doinstalowac brakujace bibliotekii.

W pliku ocr.py trzeba zmienic sciezke w ktorej mamy tesseract.exe

pytesseract.pytesseract.tesseract_cmd = r"SCIEZKA_DO_PROJEKTU\System_Klasyfikacji_Dokumentow\Tesseract\tesseract.exe"

-----

## III.  Sposob uruchomienia aplikacji

Po zainstalowaniu bibliotek oraz zmiany sciezki dla biblioteki tesseract uruchamiamy program przez odpalenie plik ui.py

-----

## VI. Struktura katalogow

SYSTEM KLASYFIKACJI DOKUMENTOW
    - dane
        - test
            - cv
            - dowody
            - faktury
            - paragony
            - wizytowki
        - trening
            - cv
            - dowody
            - faktury
            - paragony
            - wizytowki
    - Kod
        - ekstrakcja_cech.py
        - klasyfikacja.py
        - ocr.py
        - ui.py
    - odczytane
    - Tesseract
    - wyniki
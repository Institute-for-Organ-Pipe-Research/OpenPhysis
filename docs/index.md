[Aktualności](./blog/aktualnosci.md)

# Dokumentacja: Badawcza implementacja technologii Viscount Physis (US7442869B2)

## Cel projektu

Celem projektu jest zaimplementowanie fizycznego modelu dźwięku organowego zgodnego z patentem **US7442869B2**, stosowanym m.in. w technologii **Viscount Physis™**. Implementacja ta umożliwia analizę struktury modelu, generowanie tonu organowego oraz eksperymentowanie z parametrami modelu.

---

## Struktura główna

### `PhysicalModelOrgan`

Główna klasa odpowiedzialna za renderowanie dźwięku nuty z wykorzystaniem trzech komponentów:

* `HarmonicGenerator` – komponent bazowy
* `NoiseGenerator` – model generowania szumu
* `LinearResonator` – rezonator i filtracja

Działa na bazie parametrów `params` i domyślnej częstotliwości próbkowania 44100 Hz.

### `default_params()`

Zwraca słownik parametrów zgodny z wartościami domyślnymi opisanymi w patencie. Możliwa jest aktualizacja dowolnego parametru, np. `GAIN1`, `NGAIN`, `TFBK`, `attack_time` itd.

---

## Komponenty syntezy

### `HarmonicGenerator`

Odpowiada za podstawowy sygnał dźwiękowy (oscylator harmoniczny, obwiednia, modulacja amplitudy, linia opóźnienia, filtr BP, funkcja nieliniowa).

### `NoiseGenerator`

Bazuje na strukturze "NOISE BOX" z filtrami dolnoprzepustowymi, losową modulacją i limitatorem prędkości zmian.

### `LinearResonator`

Implementuje rezonator akustyczny zgodny z rysunkiem Fig. 15 z patentu. Zawiera filtr dolno- i górnoprzepustowy, filtr wszechprzepustowy i linię opóźnienia z regulowanym sprzężeniem zwrotnym.

### `EnvelopeGenerator`

Obwiednia typu ADSR dla harmonicznych i oddzielna dla szumu.

### `LowFrequencyOscillator`

Wewnętrzny LFO (modulacja amplitudy i częstotliwości).

### `FrequencyModulator`

Dodaje subtelną losową modulację częstotliwości na poziomie cyklu.

### `RateLimiter`

Ogranicza prędkość zmian sygnału (dla szumu) zgodnie z Fig. 12.

---

## Wymagania

* Python 3.8+
* NumPy
* SciPy
* ...

## Planowane rozszerzenia

* Interfejs użytkownika do eksploracji parametrów
* Uczenie maszynowe doboru parametrów na podstawie dostarczonej próbki dźwięku oczyszczonej z pogłosu
* Symulacja strojenia i przestrzennego rozmieszczenia piszczałek

---

## Licencja

Projekt edukacyjny. Brak powiązań z producentem Viscount. 

Dokładna analiza oscylatorów cyfrowych [oscylatory cyfrowe](./oscylator-cyfrowy.md)
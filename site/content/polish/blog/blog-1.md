---
title: Obecny model uczenia maszynowego nie działa. Co z nim jest nie tak?
date: 2025-08-04T16:00:00+02:00
---

W trakcie analizy kodu uczenia maszynowego zauważyłam kilka kluczowych problemów, które mogą wpływać na jakość wyników:

- brak eksportu fazy harmonicznych (ang. *harmonic phase*),
- zbyt duże poleganie na parametrach MCFF (*Monte Carlo Force Field*),
- brak przekształcenia obwiedni ADSR (*Attack, Decay, Sustain, Release*) do zestawu funkcji opisujących jej kształt.

Uczenie maszynowe zaczyna dobierać parametry spoza logicznych zakresów, co skutkuje brakiem spójnych zależności między zmiennymi. Jest to sygnał do pełnego przeglądu kodu i wyeliminowania punktów krytycznych.

Jednym z parametrów możliwych do wyprowadzenia analitycznie jest właśnie obwiednia ADSR – była ona już wcześniej zaimplementowana, jednakże uzyskane dane powinny zostać przekształcone do zestawu funkcji opisujących jej dokładny przebieg. Pozwoli to na uproszczenie procesu uczenia i ograniczenie zbędnych stopni swobody.

Im więcej takich miejsc zostanie ujednoznacznionych i wzmocnionych logiczną strukturą, tym większa szansa na uzyskanie lepszych, stabilniejszych wyników modelowania.

Zauważyłam również, że model syntezy dźwięku musi dynamicznie przyjmować parametry – nie tylko z myślą o przetwarzaniu *real-time* czy symulacji fizycznych zjawisk akustycznych. W szczególności, faza *Attack* (atak) w obwiedni ADSR może wymagać aktualizacji w czasie trwania dźwięku.

W praktyce:

- rzeczywista piszczałka organowa nie zawsze od razu generuje częstotliwość podstawową,
- początkowe oscylacje często zawierają chaotyczny lub niestabilny przebieg, który dopiero po ułamku sekundy stabilizuje się do tonu zasadniczego,
- model powinien mieć możliwość dynamicznej adaptacji do tej zmienności – np. przez reanalizę parametrów w krótkim buforze czasowym.

Wprowadzenie dynamicznych komponentów do fazy *Attack* może poprawić realizm syntezowanego dźwięku i zbliżyć go do zachowania rzeczywistych instrumentów akustycznych.

Zabiegiem, który może znacząco usprawnić proces analizy i przetwarzania danych, będzie utworzenie słownika – np. w postaci dwuwymiarowej tablicy `numpy` – w której klucz odnosi się do kodu dźwięku MIDI, a wartość do odpowiadającej mu częstotliwości podstawowej dźwięku.

Dobrze przygotowany słownik częstotliwości, sprzężony z funkcjami analitycznymi (np. obwiednią ADSR) i logicznie wyznaczonymi ograniczeniami parametrów, może znacząco podnieść jakość predykcji modelu i ogólną stabilność procesu uczenia.

Warto podkreślić, że odchylenia względem słownika częstotliwości MIDI nie stanowią problemu – są one wręcz naturalne przy analizie rzeczywistych próbek dźwięku. Istotne jest jednak, aby w ramach przygotowania danych wejściowych dokonywać przekształcenia sygnału do najbliższej częstotliwości podstawowej ze słownika.

Taka operacja służy:

- normalizacji parametrów uczenia maszynowego,
- uproszczeniu porównań między próbkami,
- ograniczeniu wpływu mikroodchyleń strojenia (np. wynikających z temperacji lub ekspresji wykonawczej),
- ułatwieniu przypisania klas dźwięku (MIDI ↔ częstotliwość).

Dzięki temu proces treningu sieci może być bardziej stabilny, a model unika niepożądanej nadwrażliwości na nieregularności sygnału.

Obecnie struktura cech wyglądała następująco:

| Obwiednia | MFCC | Chroma | Inne cechy |
| --------- | ---- | ------ | ---------- |
| Attack, Decay, Sustain, Release | 13   | 12     | spectral centroid, spectral bandwidth, zero crossing, RMS |

> [!NOTE]
> Jak widać, model uczenia maszynowego nie porównuje wygenerowanego syntetycznego dźwięku poprzez rozkład harmonicznych (*harmonic decomposition*).  
>
> Obecny model **nie mógł działać prawidłowo** – cechy takie jak **MFCC** (*Mel-Frequency Cepstral Coefficients*) **nie dostarczają pełnych informacji o strukturze dźwięku!*

MFCC dobrze sprawdzają się w rozpoznawaniu mowy, ale w kontekście syntezy instrumentów – zwłaszcza piszczałek organowych – pomijają wiele istotnych aspektów, takich jak:

- pełne widmo harmoniczne (amplituda i faza),
- nieliniowe zmiany w czasie,
- charakterystyka ataku i zaniku.

> [!Conclusion]
> Rozwarzam przejście do pełniejszej reprezentacji widmowej – np. harmonicznego rozkładu amplitud i faz lub bezpośredniego porównania z użyciem STFT.

Już w fazie wyznaczenia punktów obwiedni można natrafić na trudności. Koniec fazy ataku jest zbyt wcześnie obliczana. Obrazuje to wykres. Dzieje się tak próbując wyznaczyć koniec fazy ataku za pomocą określonego poziomu aplitudy np. threshold=0.9 i nie badając pochochodnych i wypłaszczenia otzrymuje się nie prawidłowe wyniki. Szukając nie tylko pierwszego spadku pochodnej,
ale maksimum obwiedni i jej ustabilizowania koniec fazy ataku jest prawidłowy.

![Wykres przedstawiający wyznaczony koniec fazy ataku próbki w zależności od metody](/images/blog-1-fazy-adsr.png)

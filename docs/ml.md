# Analiza cech dla syntezy dźwięku organowego

## 1. Parametry ADSR (4 cechy)

| Indeks | Nazwa          | Zakres wartości | Znaczenie dla syntezy organów                                  |
|--------|----------------|-----------------|---------------------------------------------------------------|
| 0      | Attack time    | 0.01+ sekund    | Czas narastania dźwięku (im krótszy, tym bardziej "ostry" atak) |
| 1      | Decay time     | 0.01+ sekund    | Czas przejścia do sustain (ważny dla "miękkiego" przejścia)    |
| 2      | Sustain level  | 0.0-1.0         | Poziom stałej części dźwięku (kluczowy dla głośności)          |
| 3      | Release time   | 0.01+ sekund    | Czas zanikania po zwolnieniu klawisza (efekt "echo")           |

**Przykład dla organów**:

- Długi release (~0.5s) symuluje naturalny zanik dźwięku w kościelnej akustyce
- Krótki attack (<0.05s) daje charakterystyczny "uderzeniowy" początek

## 2. MFCC (26 cech)

| Zakres indeksów | Typ cechy      | Liczba cech | Znaczenie                                                      |
|-----------------|----------------|-------------|----------------------------------------------------------------|
| 4-16            | Średnie MFCC   | 13          | Główna charakterystyka widmowa (analiza "kształtu" dźwięku)    |
| 17-29           | Deltas MFCC    | 13          | Dynamika zmian widma w czasie                                  |

**Dlaczego ważne dla organów?**

- **MFCC 1-3**: Określają ogólną "ciemność/jasność" barwy
- **MFCC 4-6**: Wpływają na postrzeganą "bogatość" brzmienia
- **Wyższe MFCC**: Oddają charakterystyczne rezonanse piszczałek

> **Uwaga**: Deltas MFCC szczególnie istotne dla fazy ataku, gdzie widmo szybko ewoluuje.

## 3. Cechy spektralne (3 cechy)

| Indeks | Nazwa                | Znaczenie                                                      |
|--------|----------------------|----------------------------------------------------------------|
| 30     | Spectral centroid    | Średnia ważona częstotliwość (im wyższa, tym "jaśniejszy" dźwięk) |
| 31     | Spectral bandwidth   | Rozpiętość widma (szersze = bardziej "szumowe")                |
| 32     | Harmonic ratio       | Stosunek energii harmonicznej do całkowitej (0-1, wyższe = bardziej czysty ton) |

**Dla syntezy organów**:

- Wysoki harmonic ratio (>0.8) = typowe dla czystych piszczałek labialnych
- Niski centroid + wąska bandwidth = charakterystyczne dla głębokich rejestrów 16'

## Związek z patentem US7442869

### ADSR

- Bezpośrednio odpowiada za generowanie obwiedni w bloku harmonicznym (patent: Fig. 7)
- Kontroluje parametry generatorów envelope (20a, 20b) z Fig. 3 patentu

### MFCC

- Odzwierciedlają efekt nieliniowych transformacji (blok 15, 26) i filtrów (27) z Fig. 3
- Przechwytują charakterystykę harmoniczną opisaną w sekcji [0057] patentu

### Cechy spektralne

- Spectral centroid związany z działaniem filtrów (47, 49) z Fig. 15
- Harmonic ratio odpowiada za balans między komponentem harmonicznym a szumowym (bloki 9 vs 11)

## Optymalizacja dla syntezy organów

**Dobre praktyki**:

```python
# Przykład ważenia cech dla organów
feature_weights = np.array([
    1.0,  # attack 
    0.7,  # decay  
    1.2,  # sustain (najważniejsze)
    0.5,  # release
    *np.ones(13),  # MFCC
    *np.linspace(1.5, 0.5, 13),  # Deltas (ważniejsze niższe częstotliwości)
    0.8,  # centroid
    1.1,  # bandwidth  
    1.5   # harmonic ratio (kluczowe)
])
```

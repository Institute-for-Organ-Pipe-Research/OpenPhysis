---
title: Silnik Physis. Czego w nim brakuje?
date: 2025-08-09T20:00:00+02:00
---

Pomimo dużej zgodności obecna implementacja silnika Physis nie wpełni odzwierciedla implementacje z patentu.

Brakujące elementy (w porównaniu do patentu):

- Generator szumu ("NOISE BOX"):

    > Patent opisuje złożony generator szumu z pętlą sprzężenia zwrotnego i ogranicznikiem ("RATE LIMITER" – Fig. 11–12).

    **W kodzie**: Brak implementacji tej funkcjonalności. Kluczowe parametry (np. NGAIN, NBFBK) są zadeklarowane, ale nieużywane.

- Rezonator nieliniowy:

    > Patent wykorzystuje zamkniętą pętlę z filtrami (Fig. 15), w tym all-pass (blok 52) do modelowania fizyki rury organowej.

    **W kodzie**: Rezonator jest uproszczony (brak filtrów i pętli sprzężenia zwrotnego). Parametr FBK istnieje, ale nie jest wykorzystywany w algorytmie.

- Modulacje częstotliwości i amplitudy:

    > Patent używa LFO (Fig. 8) do modulacji (sygnały 23 i 33).

    **W kodzie**: Brak implementacji modulacji w czasie rzeczywistym.

- Obwiednia ADSR:

    > Patent precyzyjnie kontroluje obwiednię dla faz attack/decay/sustain/release (Fig. 7).

    **W kodzie**: Parametry obwiedni (np. attack_time) są zadeklarowane, ale nie są używane w metodzie process_block().

Brak tej implementacji znacząco wpływa na nieprawidłowy dobór parametrów przez model uczenia maszynowego.

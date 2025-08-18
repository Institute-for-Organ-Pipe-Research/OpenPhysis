---
title: Architektura
date: 2025-07-30T17:00:00+02:00
---

## Planowany podział projektu

W obecnej fazie wszystko znajduje się w jednym repozytorium. W przyszłości projekt zostanie podzielony na moduły:

| Repozytorium (planowane) | Opis                                                          |
| ------------------------ | ------------------------------------------------------------- |
| `OpenPhysis`             | Główna aplikacja VPO, interfejs, konfiguracja instrumentu     |
| `fonus-core`             | Silnik DSP i system syntezy widmowej                          |
| `fonus-ml`               | Moduły uczenia maszynowego do przewidywania parametrów głosów |
| `fonus-fdtd`             | Symulacje fizyczne (FDTD) i dane menzuracyjne                 |
| `physis-docs`            | Dokumentacja techniczna i badawcza projektu                   |

## Stan obecny (pre-alfa)

- [x] Prototyp silnika syntezy wzorowanego na Viscount Physis (offline)
- [x] Interaktywny generator komponentów harmonicznych (Jupyter)
- [ ] Refaktoryzacja kodu
- [ ] Rozbudowa silnika do działania w czasie rzeczywistym
- [ ] Integracja z systemem ML i FDTD

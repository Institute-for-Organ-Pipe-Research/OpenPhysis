# OpenPhysis – powered by FONUS

**Wersja:** pre-alpha  
**Status:** Wczesny prototyp (eksperymentalny, wciąż bez refaktoryzacji)

## Cel projektu

**OpenPhysis** to otwarty projekt cyfrowych organów piszczałkowych (VPO), zaprojektowany z myślą o maksymalnym realizmie brzmienia, elastyczności i kreatywnym podejściu do instrumentu.

> _"Physis – an open framework for spectral pipe voice modeling"_

## Czym jest OpenPhysis?

- Symulator organów cyfrowych, inspirowany Viscount Physis, ale tworzony od podstaw jako **otwarty projekt**.
- Narzędzie dla użytkowników zaawansowanych i badaczy chcących tworzyć własne instrumenty, a nie tylko korzystać z gotowych presetów, sampli.
- Projekt oparty na **fizycznym modelowaniu**, **analizie widma** i **uczeniu maszynowym**.

## Planowany podział projektu

W obecnej fazie wszystko znajduje się w jednym repozytorium. W przyszłości projekt zostanie podzielony na moduły:

| Repozytorium (planowane) | Opis |
|--------------------------|------|
| `OpenPhysis`             | Główna aplikacja VPO, interfejs, konfiguracja instrumentu |
| `fonus-core`             | Silnik DSP i system syntezy widmowej |
| `fonus-ml`               | Moduły uczenia maszynowego do przewidywania parametrów głosów |
| `fonus-fdtd`             | Symulacje fizyczne (FDTD) i dane menzuracyjne |
| `physis-docs`            | Dokumentacja techniczna i badawcza projektu |

## Stan obecny (pre-alfa)

- [x] Prototyp silnika syntezy wzorowanego na Viscount Physis (offline)
- [x] Interaktywny generator komponentów harmonicznych (Jupyter)
- [-] Refaktoryzacja kodu
- [-] Rozbudowa silnika do działania w czasie rzeczywistym
- [-] Integracja z systemem ML i FDTD

## Dokumentacja

W fazie tworzenia. Notatki projektowe i eksperymenty znajdują się w folderze `notebooks/`.

Notatniki:

- [oscilator.ipynb](./notebooks/oscylator-cyfrowy.ipynb) - zgodnie z dokumentacją Viscout Physis odpowiada to blokowi 9 i 10 - patrz rys. 2 [Viscout Physis](./docs/attachments/US7442869.pdf)

## Zależności

Aby zainstalować zależności wykonaj: `pip install -r requirements.txt`

## Licencja

OpenPhysis™ is an open-source software licensed under the Mozilla Public License 2.0.
The name "OpenPhysis" and associated branding elements are reserved by the author and may not be used to promote derived works without permission.


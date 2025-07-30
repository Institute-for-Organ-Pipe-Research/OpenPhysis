---
title: OpenPhysis powerade by FONUS
date: 2025-07-30T16:00:00+02:00
draft: false
type: default
---

![](/images/openphysis-logo.webp)

> [!Note]
> **Wersja:** pre-alpha  
> **Status:** Wczesny prototyp (eksperymentalna)

## Czym jest OpenPhysis?

**OpenPhysis** to oprogramowanie wirtualnych organów piszczałkowych, zaprojektowany z myślą o maksymalnym realizmie brzmienia, elastyczności i kreatywnym podejściu do instrumentu. Oparty jest o modelowanie fizyczne a nie sample jak w Hauptwerk czy GrandOrgue. Silnik syntezy dźwięku _**FONUS**_ oparty jest o algorytm pochodzący z patentu Viscount International [US7442869B2](https://patents.google.com/patent/US7442869B2/en) dostępnym w domenie publicznej. Zapewnia to maksymalny realizm brzmienia.

## Czym rózni się OpenPhysis od innego oprogramowania VPO?

**OpenPhysis** nie odtwarza on próbek tak jak GrandOrgue czy Hauptwerk[^1]. Oparty jest o modelowanie fizyczne.

**OpenPhysis**:

- nie odtwarza próbek,
- modeluje zjawiska fizyczne w piszczałce i wiatrownicy,
- pozwala kreować barwę głosów organowych poprzez edytor.

Silnik syntezy _**FONUS**_ ma wbudowane uczenie maszynowe, które ułatwia dobór parametrów syntezy. Interakcje są niedeterministyczne i trudno przewidzeć wpływ jednego parametru na inne. Dzięki uczeniu maszynowemu będzie mozna kreować barwę za pomocą kilku parametrów analogicznych do menzury piszczałki.

## Czym jest modelowanie fizyczne?

Modelowanie fizyczne są metodami opisu matematycznego zjawisk wizycznych wystepujacych np. w piszczałce organowej. Jest wiele metod modelow ania fizycznego. Wiele z nich wymaga dużej mocy obliczeniowej i nie jest w stanie działać w czasie rzeczywistym. Viscount opracował uproszczony model, który eliminuje wady modelowania fizycznego przez falowód (waveguide synthesis). Przystępne opracowanie algorytmu, stworzył dr Colin Pykett [Viscount Organs - some observations on physical modelling patent US7442869](https://www.colinpykett.org.uk/physical-modelling-viscount-organ-patent.htm).

## Przypisy

[^1]: Hauptwerk potrafi wpływać na próbki. Ma zaimplementowany _model wiatru_: rozstrajanie, tremulant oraz tłumienie szafy ekspesyjnej.

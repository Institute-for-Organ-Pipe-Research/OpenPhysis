---
title: OpenPhysis powerade by FONUS
date: 2025-07-30T16:00:00+02:00
draft: false
type: default
---

![](/images/openphysis-logo.webp)

> [!Note]
> **Version:** pre-alpha  
> **Status:** Early prototype (experimental)

## What is OpenPhysis?

**OpenPhysis** is virtual pipe organ software designed for maximum sound realism, flexibility, and a creative approach to the instrument. It is based on physical modeling rather than samples like Hauptwerk or GrandOrgue. The sound synthesis engine _**FONUS**_ is based on an algorithm from a Viscount International patent [US7442869B2](https://patents.google.com/patent/US7442869B2/en), which is now in the public domain. This ensures maximum sound realism.

## How does OpenPhysis differ from other VPO software?

**OpenPhysis** does not play back samples like GrandOrgue or Hauptwerk[^1]. It is based on physical modeling.

**OpenPhysis**:

- does not use samples,
- models physical phenomena in the pipe and windchest,
- allows creation of organ stop timbres via an editor.

The _**FONUS**_ synthesis engine has built-in machine learning that facilitates the tuning of synthesis parameters. Interactions are nondeterministic and it is difficult to predict the influence of one parameter on others. Thanks to machine learning, it will be possible to create timbre using just a few parameters analogous to pipe scaling (mensuration).

## What is physical modeling?

Physical modeling refers to mathematical methods describing physical phenomena occurring, for example, in an organ pipe. There are many physical modeling methods. Many require high computational power and cannot run in real time. Viscount developed a simplified model that eliminates the drawbacks of physical modeling by waveguide synthesis. An accessible description of the algorithm was created by Dr. Colin Pykett [Viscount Organs - some observations on physical modelling patent US7442869](https://www.colinpykett.org.uk/physical-modelling-viscount-organ-patent.htm).

## Notes

[^1]: Hauptwerk can influence samples. It has an implemented _wind model_: detuning, tremulant, and swell box attenuation.

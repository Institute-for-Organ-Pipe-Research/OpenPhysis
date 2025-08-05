---
title: Current Machine Learning Model Does Not Work. What Is Wrong With It?
date: 2025-08-04T16:00:00+02:00
---

During the analysis of the machine learning code, I noticed several key issues that may affect the quality of the results:

- lack of export of harmonic phase,
- excessive reliance on MCFF (*Monte Carlo Force Field*) parameters,
- absence of transformation of the ADSR (*Attack, Decay, Sustain, Release*) envelope into a set of descriptive functions characterizing its shape.

The machine learning starts to select parameters outside logical ranges, resulting in no consistent relationships between variables. This signals the need for a full code review and elimination of critical points.

One parameter that can be analytically derived is the ADSR envelope – it was previously implemented, but the obtained data should be transformed into a set of functions describing its precise course. This will simplify the learning process and reduce unnecessary degrees of freedom.

The more such parts are unambiguously defined and reinforced by logical structure, the higher the chance of achieving better, more stable modeling results.

I also noticed that the sound synthesis model must dynamically accept parameters – not only with real-time processing or physical acoustic simulation in mind. In particular, the *Attack* phase of the ADSR envelope may require updating during the sound duration.

In practice:

- a real organ pipe does not always immediately produce the fundamental frequency,
- initial oscillations often contain chaotic or unstable behavior that only stabilizes into the fundamental tone after a fraction of a second,
- the model should be able to dynamically adapt to this variability – for example, by reanalyzing parameters in a short time buffer.

Introducing dynamic components into the *Attack* phase could improve the realism of the synthesized sound and bring it closer to the behavior of real acoustic instruments.

An operation that can significantly improve the data analysis and processing is creating a dictionary – e.g., as a two-dimensional `numpy` array – where the key corresponds to the MIDI sound code, and the value corresponds to the fundamental frequency of that sound.

A well-prepared frequency dictionary, coupled with analytical functions (e.g., ADSR envelope) and logically determined parameter constraints, can greatly enhance model prediction quality and overall training stability.

It is important to emphasize that deviations from the MIDI frequency dictionary are not a problem – they are natural when analyzing real sound samples. However, it is essential that input data preparation includes mapping the signal to the closest fundamental frequency from the dictionary.

Such an operation serves:

- normalization of machine learning parameters,
- simplification of comparisons between samples,
- limiting the impact of micro-tuning deviations (e.g., from temperament or expressive performance),
- facilitating assignment of sound classes (MIDI ↔ frequency).

Thanks to this, the network training process can be more stable, and the model avoids unwanted sensitivity to signal irregularities.

Currently, the feature structure looked like this:

| Envelope                      | MFCC | Chroma | Other Features                          |
| ----------------------------- | ---- | ------ | ------------------------------------- |
| Attack, Decay, Sustain, Release | 13   | 12     | spectral centroid, spectral bandwidth, zero crossing, RMS |

> [!NOTE]
> As can be seen, the machine learning model does not compare the generated synthetic sound by harmonic decomposition.  
>
> The current model **could not work properly** – features like **MFCC** (*Mel-Frequency Cepstral Coefficients*) **do not provide complete information about the sound structure!**

MFCCs work well in speech recognition but in the context of instrument synthesis – especially organ pipes – they omit many important aspects such as:

- full harmonic spectrum (amplitude and phase),
- nonlinear changes over time,
- characteristics of attack and decay.

> [!Conclusion]
> I am considering moving to a fuller spectral representation – e.g., harmonic amplitude and phase decomposition or direct comparison using STFT.

Already at the stage of determining envelope points, difficulties may arise. The attack phase end is computed too early. This is shown in the plot. This happens when trying to find the attack end using a fixed amplitude level, e.g., threshold=0.9, without examining derivatives and flattening — it gives incorrect results. By searching not only for the first derivative drop but for the envelope maximum and its stabilization, the attack phase end is correctly determined.

![Plot showing detected attack phase end of the sample depending on the method](/images/blog-1-fazy-adsr.png)

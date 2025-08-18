---
title: Architecture
date: 2025-07-30T17:00:00+02:00
---

## Planned Project Structure

Currently, everything is in one repository. In the future, the project will be divided into modules:

| Repository (planned)      | Description                                      |
|--------------------------|------------------------------------------------|
| `OpenPhysis`             | Main VPO application, interface, instrument configuration |
| `fonus-core`             | DSP engine and spectral synthesis system       |
| `fonus-ml`               | Machine learning modules for predicting stop parameters |
| `fonus-fdtd`             | Physical simulations (FDTD) and scaling data   |
| `physis-docs`            | Technical and research documentation of the project |

## Current Status (pre-alpha)

- [x] Prototype of synthesis engine modeled on Viscount Physis (offline)
- [x] Interactive harmonic component generator (Jupyter)
- [ ] Code refactoring
- [ ] Expansion of engine for real-time operation
- [ ] Integration with ML and FDTD systems

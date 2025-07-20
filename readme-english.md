# OpenPhysis – powered by FONUS

**Version:** pre-alpha  
**Status:** Early prototype (experimental, not yet refactored)

## Project Goal

**OpenPhysis** is an open-source virtual pipe organ (VPO) project, designed with a focus on maximum realism, flexibility, and a creative approach to the instrument.

> _"Physis – an open framework for spectral pipe voice modeling"_

## What is OpenPhysis?

- A digital organ simulator inspired by Viscount Physis, but built from scratch as an **open project**.
- A tool for advanced users and researchers who want to build their own instruments, not just use presets or samples.
- Based on **physical modeling**, **spectral analysis**, and **machine learning**.

## Planned Project Structure

Currently, everything resides in a single repository. In the future, the project will be split into modules:

| Repository (planned)    | Description |
|-------------------------|-------------|
| `OpenPhysis`            | Main VPO application, interface, instrument configuration |
| `fonus-core`            | DSP engine and spectral synthesis system |
| `fonus-ml`              | Machine learning modules for pipe voice parameter prediction |
| `fonus-fdtd`            | Physical simulations (FDTD) and mensuration data |
| `physis-docs`           | Technical and research documentation for the project |

## Current Status (pre-alpha)

- [x] Prototype of synthesis engine inspired by Viscount Physis (offline)
- [x] Interactive harmonic component generator (Jupyter)
- [ ] Code refactoring
- [ ] Engine extension for real-time operation
- [ ] Integration with ML and FDTD systems

## Documentation

Currently under development. Project notes and experiments can be found in the `notebooks/` folder.

Notebooks:

- [oscilator.ipynb](./notebooks/oscylator_cyfrowy.ipynb) – according to Viscount Physis documentation, corresponds to blocks 9 and 10 – see Fig. 2 in [Viscount Physis](./docs/attachments/US7442869.pdf)

## Dependencies

To install dependencies, run:  
`pip install -r requirements.txt`

## License

OpenPhysis™ is an open-source software licensed under the Mozilla Public License 2.0.  
The name "OpenPhysis" and associated branding elements are reserved by the author and may not be used to promote derived works without permission.

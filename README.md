# LSTM-NN-multidirectional
This repository contains the code, configuration, and minimal sample data to train/infer a **time-domain LSTM surrogate** that predicts:
- **6-DoF platform motions** (surge, sway, heave, roll, pitch, yaw), and  
- **fairlead/mooring tensions** at the three fairleads,
for the **OC3 spar** under **short-crested seas**.  
Directional spreading is represented by **cosine-power** distributions discretized via the **equal-energy** method into a finite set of incident headings.  
The surrogate is trained on a synthetic database of **OpenFAST** simulations and then used for fast predictions (fatigue-grade post-processing included).

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---
**Key contributions** (high-level):
- Early-fusion **LSTM** that ingests the per-direction **wave elevation time series** for Nd headings (plus wind channels) and outputs coupled motions + fairlead tensions.
- **Short-crested seas** synthesized with cosine-power spreading, discretized by **equal-energy** quadrature into **Nd** representative directions (typically Nd≈5).
- **SeaState I/O tweak** to output **per-direction** free-surface elevation time series (besides the composite), enabling supervised training with directional inputs.
- **Fatigue workflow** (rainflow + Miner on mooring stress) to compare OpenFAST vs. NN predictions on **annual damage** and **DEL** consistency.

---
## Documentation
Previous articles from the authors

---
### Modified dependencies
- **OpenFAST – SeaState (modified)** under Apache-2.0.
  Code included in `third_party/openfast-seastate-mod` with licence preserved.

# LSTM-NN-multidirectional
This repository contains the code, configuration, and minimal sample data to train/infer a **time-domain LSTM surrogate** that predicts:
- **6-DoF platform motions** (surge, sway, heave, roll, pitch, yaw), and  
- **fairlead/mooring tensions** at the three fairleads,

for the **OC3 spar** under **short-crested seas**. 

Directional spreading is represented by **cosine-power** distributions discretized via the **equal-energy** method into a finite set of incident headings. The surrogate is trained on a synthetic database of **OpenFAST** simulations and then used for fast predictions (fatigue-grade post-processing included).

The repository is contains:
- The main LSTM network code with additional datasets in the folder `third_party`
- The modified version of the Seastate source code in `third_party`

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)


## Documentation
Previous articles from the authors:

Medina-Manuel, A.; Molina Sánchez, R.; Souto-Iglesias, A. **AI-Driven Model Prediction of Motions and Mooring Loads of a Spar Floating Wind Turbine in Waves and Wind.** *Journal of Marine Science and Engineering* **2024**, 12(9), 1464. https://doi.org/10.3390/jmse12091464


## Modified dependencies
- **OpenFAST – SeaState (modified)** under Apache-2.0.
  Code included in `third_party/openfast-seastate-mod` with licence preserved.

---
### Author profiles
- ResearchGate: https://www.researchgate.net/profile/Antonio-Medina-Manuel/research
- Google Scholar: https://scholar.google.es/citations?hl=es&user=bHgb4XYAAAAJ

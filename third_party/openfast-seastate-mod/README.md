# OpenFAST SeaState (modified)

This directory contains **modifications** to the OpenFAST **SeaState** module.

- **Source**: https://github.com/OpenFAST/openfast (path: `modules/seastate`)
- **Original commit/tag**: `v4.1.2` (`40b94af`) 
- **Original licence**: Apache-2.0 (see `LICENSE-OPENFAST`)
- **Changeset**: see `NOTICE`

## What has been modified?
- Directional discretisation (equal-energy) / short seas.
- Normalisation of D(θ) consistent with F(θ)=∫D(θ)dθ.
- Minor corrections to I/O and default parameters.
*(Adjust this list to your actual changes.)*

## Use
These files are a modified copy to accompany the article.
Also cite OpenFAST as indicated.

## Recommended citation
- Software from this repo: see `CITATION.cff` in the root directory.
- OpenFAST: cite as original software; documentation and Apache-2.0 licence.

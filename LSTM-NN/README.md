# LSTM Model for FOWTs inference
This directory contains the source code for reproducibility of an early fusion surrogate model to predict FOWT responses from wave time series.
## Features
- HDF5 I/O compatible with OpenFAST-style exports
- Min–max normalization via JSON (same stats used during training)
- Sequence-to-sequence or sequence-to-one prediction
- Optional evaluation & CSV export

## Quickstart

```bash
main_fowt_lstm.py \python predict_fowt_lstm.py \
  --inputs-h5 data/inputs.h5 \
  --outputs-h5 data/outputs.h5 \
  --dof heave \
  --norm-json configs/normalization.json \
  --checkpoint models/heave_epoch20.pt \
  --out-h5 predictions.h5 \
  --multi-dir --dir-num 5 \
  --interp-N 40 \
  --seq-len 80 \
  --batch-size 64 \
  --hidden-size 128 \
  --num-layers 5 \
  --dropout 0.3 \
  --nn-pyc private_bin/nn_models.cpython-310.pyc

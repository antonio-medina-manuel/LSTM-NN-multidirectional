# LSTM Model for FOWTs inference
This directory contains the source code for reproducibility of an early fusion surrogate model to predict FOWT responses from wave time series.
## Features
- HDF5 I/O compatible with SONG/OpenFAST-style exports
- Min–max normalization via JSON (same stats used during training)
- Sequence-to-sequence or sequence-to-one prediction
- Optional evaluation & CSV export

## Quickstart

```bash
main_fowt_lstm.py \
  --inputs-h5 data/inputs.h5 \
  --norm-json configs/norm_relwave.json \
  --checkpoint models/relwave_seq50.pt \
  --out-h5 predictions.h5 \
  --seq-len 50 --batch-size 64 --hidden-size 64 --num-layers 5 --dropout 0.7 --seq2seq

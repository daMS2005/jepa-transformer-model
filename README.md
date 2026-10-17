# JEPA Transformer Model

A financial time-series model in two stages. A JEPA learns one representation per
asset-day from point-in-time daily features: a context encoder sees a masked view
and predicts the full-view representation of an exponential-moving-average target
encoder. The JEPA is then frozen, and a causal decoder Transformer (RoPE,
QK-RMSNorm, SwiGLU with top-k mixture-of-experts blocks, Student-t or linear head)
reads each asset's chronological representations to forecast 1-, 5- and
20-session returns. Optional Triton kernels cover the MoE experts and FP32-accurate
3xTF32 linear layers.

The configurations in `configs/` are examples: `configs/data/example.toml` is a
small feature specification that exercises every mechanism of the data layer, and
`configs/model/example.toml` documents every model and optimization field. Data,
credentials and checkpoints are kept outside this repository.

## Data

`jepa_transformer.data.build.build_cache` reads a DuckDB replica read-only and
exports features, labels and company-level splits into array shards, which
`jepa_transformer.data.dataset` serves to both models. The replica provides:

- `prices_daily_model_v1`: daily bars (`con_id`, `trading_date`, open, high, low,
  close, volume, `split_ratio`, `previous_link_valid`);
- `universe_membership`: effective-dated universe members (`universe_name`,
  `con_id`, `symbol`, `valid_from`, `valid_to`);
- `sec_security_mappings`, `sec_companies` and `securities`: company identity and
  industry, used for the company-level splits;
- any tables the feature specification's blocks read.

Each feature is SQL in the specification. Rows may use only information available
by the close of their trading day; the loader rejects forward-looking window
shapes and known look-ahead columns, fits normalization and PCA on training rows
only, and keeps missing values missing behind explicit mask channels. Companies,
not rows, are split into core, extended and holdout groups.

## Training

Edit the settings at the top of `train.py` (preset, database, cache and output
paths, stages), then run it:

```bash
python -m pip install -e '.[data]'
python train.py
```

The JEPA and the Transformer have independent optimizers and checkpoints. The
Transformer reads raw feature windows and runs them through the frozen JEPA on
the fly. Each stage validates on the validation era's holdout companies every
epoch, keeps the best epoch as `<name>.best.pt`, and stops after `patience`
epochs without improvement. The same functions (`jepa_transformer.training.run`)
can be called from a notebook.

Run the test suite with `python -m unittest discover -s tests -v`.

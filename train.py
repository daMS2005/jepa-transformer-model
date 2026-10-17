"""Build the cache, pretrain the JEPA, train the Transformer, evaluate on holdout validation.

Edit the settings below, then run ``python train.py``. The stages are plain functions
(jepa_transformer.data.build.build_cache, jepa_transformer.training.run) and can be
called the same way from a notebook.
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from jepa_transformer.training.run import evaluate, train_jepa, train_transformer  # noqa: E402

PRESET = ROOT / "configs/model/example.toml"
DATABASE = Path("/path/to/market.duckdb")
CACHE = Path("/path/to/cache/model_a")
OUT = Path("/path/to/checkpoints/example")
STAGES = ["jepa", "transformer", "evaluate"]  # put "cache" first to rebuild CACHE from DATABASE
RUN = {"device": None, "precision": None, "limit_batches": None}  # None: the preset's device/precision; limit_batches=10 smoke-tests
KERNEL_BACKEND = None  # None: the preset's ("triton", CUDA only); "torch" anywhere else

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "cache" in STAGES:
        from jepa_transformer.data.build import build_cache  # needs duckdb; only this stage does

        build_cache(ROOT / "configs/data/example.toml", DATABASE, CACHE, "A", memory_limit="3GB", threads=4)
    if "jepa" in STAGES:
        train_jepa(PRESET, CACHE, OUT, **RUN)
    if "transformer" in STAGES:
        train_transformer(PRESET, CACHE, OUT, kernel_backend=KERNEL_BACKEND, **RUN)
    if "evaluate" in STAGES:
        metrics = evaluate(CACHE, OUT, device=RUN["device"], precision=RUN["precision"])
        (OUT / "eval_val_holdout.json").write_text(json.dumps(metrics, indent=1))
        print(json.dumps(metrics["rank_ic"], indent=1))

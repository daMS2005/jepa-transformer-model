"""JEPA and Transformer training primitives."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jepa_transformer.config import TransformerConfig, load_model_config
from jepa_transformer.models.jepa import JEPA
from jepa_transformer.models.transformer import Transformer
from jepa_transformer.training.checkpoint import load_checkpoint, save_checkpoint
from jepa_transformer.training.jepa import build_jepa, latent_loss, train_jepa_epoch
from jepa_transformer.training.transformer import (
    evaluate_transformer,
    forecast_loss,
    train_transformer_epoch,
)


PRESET = Path(__file__).resolve().parents[1] / "configs" / "model" / "example.toml"
SMALL = TransformerConfig(
    input_dim=4, sequence_lengths=(3,), target_dim=1, learning_rate=1e-3, num_epochs=1, jepa_checkpoint="jepa.pt",
    d=8, out_proj=8, n_heads=1, n_layers=4, hidden_dim=16, max_seq=8, moe_layers=(1, 2), n_experts=2, head="student_t",
)
GROUPS = (("price", 4), ("market", 1), ("fundamentals", 3), ("macro", 2), ("sec_descriptions", 10))


class JEPAConfigurationTest(unittest.TestCase):
    def test_feature_groups_do_not_enable_unrequested_stems(self) -> None:
        jepa = replace(load_model_config(PRESET).jepa, input_dim=20, feature_groups=GROUPS)
        model = build_jepa(jepa)
        self.assertEqual(len(model.encoder.input_encoders), 0)
        self.assertEqual(model.encoder.net[0].in_features, 20)


class MaskingAndLossTest(unittest.TestCase):
    def test_description_components_are_masked_as_one_block(self) -> None:
        torch.manual_seed(3)
        model = JEPA(6, 4, hidden_dim=8, depth=1, dropout=0.0, mask_units=(1, 1, 4), mask_ratio=0.5)
        result = model(torch.ones(128, 6))
        description_mask = result.feature_mask[:, 2:]
        self.assertTrue(((description_mask == description_mask[:, :1]).all(dim=1)).all())
        self.assertTrue(description_mask.any() and (~description_mask).any())

    def test_losses_reject_unlabelled_or_unmaskable_batches(self) -> None:
        with self.assertRaises(ValueError):
            latent_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            forecast_loss(torch.zeros(1, 2, 1), torch.zeros(1, 2, 1), torch.zeros(1, 2, dtype=torch.bool))

    def test_student_t_forecast_loss_is_finite(self) -> None:
        model = Transformer(SMALL)
        result = model(torch.randn(2, 2, 4))
        loss = forecast_loss(result.pred, torch.randn(2, 2, 1), torch.ones(2, 2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))


class CheckpointTest(unittest.TestCase):
    def test_atomic_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, {"format_version": 1, "value": torch.tensor([1.0])})
            self.assertEqual(load_checkpoint(path)["value"].item(), 1.0)


class TrainingWorkflowTest(unittest.TestCase):
    def test_small_models_train_and_evaluate(self) -> None:
        jepa = JEPA(4, 4, hidden_dim=8, depth=1, dropout=0.0)
        optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)
        jepa_loss = train_jepa_epoch(
            jepa, optimizer, [torch.randn(8, 4)], device=torch.device("cpu"),
        )
        self.assertGreater(jepa_loss, 0)

        transformer = Transformer(SMALL)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
        batches = DataLoader(TensorDataset(
            torch.randn(4, 3, 4), torch.randn(4, 3, 1), torch.ones(4, 3, dtype=torch.bool),
        ), batch_size=2)
        loss = train_transformer_epoch(
            transformer, optimizer, batches, device=torch.device("cpu"),
        )
        metrics = evaluate_transformer(transformer, batches, device=torch.device("cpu"))
        self.assertTrue(np.isfinite(loss) and np.isfinite(metrics["loss"]))
        self.assertEqual(len(metrics["mae_by_horizon"]), 1)

if __name__ == "__main__":
    unittest.main()

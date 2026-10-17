"""Model-preset and JEPA residual-stream contracts."""

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from jepa_transformer.config import ModelConfig, load_model_config
from jepa_transformer.models.jepa import JEPA


ROOT = Path(__file__).parents[1]


class ModelPresetTests(unittest.TestCase):
    def test_annotated_example_is_valid(self) -> None:
        load_model_config(ROOT / "configs" / "model" / "example.toml")

    def test_cross_model_width_mismatch_is_rejected(self) -> None:
        config = load_model_config(ROOT / "configs" / "model" / "example.toml")
        with self.assertRaisesRegex(AssertionError, "JEPA output"):
            ModelConfig(
                runtime=config.runtime,
                jepa=replace(config.jepa, output_dim=512),
                transformer=config.transformer,
            )


class JEPAResidualTests(unittest.TestCase):
    def test_residual_encoder_preserves_public_output_shape(self) -> None:
        model = JEPA(
            input_dim=12,
            output_dim=8,
            hidden_dim=16,
            depth=3,
            dropout=0.0,
            residual=True,
        )
        self.assertEqual(model.encode(torch.randn(4, 12)).shape, (4, 8))


if __name__ == "__main__":
    unittest.main()

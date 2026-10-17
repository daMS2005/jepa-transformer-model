"""Validated runtime and model-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution settings that do not define either model's learning task."""

    seed: int = 0
    device: str = "cpu"
    precision: str = "float32"
    num_workers: int = 0
    checkpoint_directory: str = "checkpoints"

    def __post_init__(self) -> None:
        assert self.seed >= 0, "seed must be non-negative"
        assert self.device, "device must not be empty"
        assert self.precision in {"float32", "float16", "bfloat16"}, "precision must be float32, float16, or bfloat16"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.checkpoint_directory, "checkpoint_directory must not be empty"


@dataclass(frozen=True, slots=True)
class JEPAConfig:
    """JEPA architecture and optimization settings."""

    input_dim: int
    output_dim: int
    hidden_dim: int
    depth: int
    dropout: float
    mask_ratio: float
    target_momentum: float
    learning_rate: float
    num_epochs: int
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    feature_groups: tuple[tuple[str, int], ...] = ()
    input_encoder_dim: int | None = None
    input_encoder_depth: int = 1
    residual: bool = False
    mask_units: tuple[int, ...] = ()  # cache meta "mask_units"; an embedding's components mask as one unit

    def __post_init__(self) -> None:
        assert min(self.input_dim, self.output_dim, self.hidden_dim, self.depth) > 0, "JEPA dimensions and depth must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert 0.0 < self.mask_ratio < 1.0, "mask_ratio must be strictly between 0 and 1"
        assert 0.0 <= self.target_momentum < 1.0, "target_momentum must be in [0, 1)"
        assert self.learning_rate > 0.0, "learning_rate must be positive"
        assert self.num_epochs >= 1, "num_epochs must be at least 1"
        assert 0.0 <= self.min_learning_rate <= self.learning_rate, "min_learning_rate must be between zero and learning_rate"
        assert self.weight_decay >= 0.0, "weight_decay must be non-negative"
        assert self.max_grad_norm is None or self.max_grad_norm > 0.0, "max_grad_norm must be positive when provided"
        assert self.input_encoder_dim is None or self.input_encoder_dim > 0, "input_encoder_dim must be positive"
        assert self.input_encoder_depth > 0, "input_encoder_depth must be positive"
        assert not self.residual or self.depth > 1, "residual JEPA requires depth greater than one"
        assert not self.mask_units or sum(self.mask_units) == self.input_dim, "mask_units must sum to input_dim"
        if self.feature_groups:
            names = tuple(name for name, _ in self.feature_groups)
            assert len(set(names)) == len(names), "feature group names must be unique"
            assert all(name and size > 0 for name, size in self.feature_groups), "feature groups must be named and positive"
            assert sum(size for _, size in self.feature_groups) == self.input_dim, "feature groups must sum to input_dim"


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Causal forecasting architecture and its separate training settings.

    ``out_proj`` is each Q/K/V projection's width across heads, not target width.
    ``moe_layers`` contains zero-based block indices; an empty tuple is dense.
    """

    input_dim: int
    sequence_lengths: tuple[int, ...]
    target_dim: int
    learning_rate: float
    num_epochs: int
    jepa_checkpoint: str
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    d: int = 128
    out_proj: int = 128
    n_heads: int = 4
    n_layers: int = 6
    hidden_dim: int = 352
    max_seq: int = 512
    moe_topk: int = 1
    n_experts: int = 4
    moe_layers: tuple[int, ...] = (2, 3)
    moe_loss_weight: float = 0.01
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    head: str = "linear"
    kernel_backend: str = "torch"

    def __post_init__(self) -> None:
        assert min(self.input_dim, self.target_dim) > 0, "Transformer input and target sizes must be positive"
        assert self.learning_rate > 0.0, "learning_rate must be positive"
        assert self.num_epochs >= 1, "num_epochs must be at least 1"
        assert self.jepa_checkpoint, "jepa_checkpoint must not be empty"
        assert 0.0 <= self.min_learning_rate <= self.learning_rate, "min_learning_rate must be between zero and learning_rate"
        assert self.weight_decay >= 0.0, "weight_decay must be non-negative"
        assert self.max_grad_norm is None or self.max_grad_norm > 0.0, "max_grad_norm must be positive when provided"
        assert min(self.d, self.out_proj, self.n_heads, self.n_layers, self.hidden_dim, self.max_seq) > 0, "Transformer dimensions and layer counts must be positive"
        assert self.out_proj % self.n_heads == 0 and (self.out_proj // self.n_heads) % 2 == 0, "out_proj must divide into n_heads with an even head dimension for RoPE"
        assert self.sequence_lengths, "sequence_lengths must not be empty"
        assert len(set(self.sequence_lengths)) == len(self.sequence_lengths), "sequence_lengths must be unique"
        assert all(0 < length <= self.max_seq for length in self.sequence_lengths), "sequence_lengths must be positive and no greater than max_seq"
        assert len(set(self.moe_layers)) == len(self.moe_layers), "moe_layers must not contain duplicates"
        assert all(0 < idx < self.n_layers - 1 for idx in self.moe_layers), "MoE indices must be middle layers; first and last blocks stay dense"
        assert self.n_experts >= 1 and 1 <= self.moe_topk <= self.n_experts, "moe_topk must be between 1 and n_experts"
        assert self.moe_loss_weight >= 0 and 0 <= self.dropout < 1, "moe_loss_weight must be nonnegative and dropout must be in [0, 1)"
        assert self.rope_base > 1 and self.norm_eps > 0, "rope_base must exceed 1 and norm_eps must be positive"
        assert self.head in {"linear", "student_t"}, "head must be linear or student_t"
        assert self.kernel_backend in {"torch", "triton"}, "kernel_backend must be torch or triton"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One validated JEPA-to-Transformer preset."""

    runtime: RuntimeConfig
    jepa: JEPAConfig
    transformer: TransformerConfig

    def __post_init__(self) -> None:
        assert self.jepa.output_dim == self.transformer.input_dim, "JEPA output must match Transformer input"
        assert self.transformer.input_dim == self.transformer.d, "Transformer input must match its residual width"
        assert self.jepa.hidden_dim == self.jepa.output_dim, "JEPA residual width must match its output width"
        assert self.transformer.out_proj == self.transformer.d, "Q/K/V width must match Transformer width"
        assert self.transformer.out_proj // self.transformer.n_heads == 64, "attention heads must be 64 wide"
        assert self.transformer.hidden_dim == 4 * self.transformer.d, "SwiGLU hidden width must be 4x Transformer width"


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a TOML preset and validate both models together."""

    with Path(path).open("rb") as config_file:
        raw = tomllib.load(config_file)

    jepa = dict(raw["jepa"])
    transformer = dict(raw["transformer"])
    if "feature_groups" in jepa:
        jepa["feature_groups"] = tuple(tuple(group) for group in jepa["feature_groups"])
    if "moe_layers" in transformer:
        transformer["moe_layers"] = tuple(transformer["moe_layers"])
    if "sequence_lengths" in transformer:
        transformer["sequence_lengths"] = tuple(transformer["sequence_lengths"])

    return ModelConfig(
        runtime=RuntimeConfig(**raw["runtime"]),
        jepa=JEPAConfig(**jepa),
        transformer=TransformerConfig(**transformer),
    )

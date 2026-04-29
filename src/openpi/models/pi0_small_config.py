import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_small import Pi0Small


@dataclasses.dataclass(frozen=True)
class Pi0SmallConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    gemma_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # If True, freeze the SigLIP vision encoder during training. Saves ~3.2GB of optimizer
    # memory, recommended for GPUs with <=24GB VRAM.
    freeze_vision: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_SMALL

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Small":
        from openpi.models.pi0_small import Pi0Small

        return Pi0Small(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.freeze_vision:
            # Freeze SigLIP encoder params (PaliGemma/img/...) but NOT the head layer,
            # since the head (1152->1024) is freshly initialized and must be trained.
            return nnx.All(
                nnx_utils.PathRegex(".*img.*"),
                nnx.Not(nnx_utils.PathRegex(".*img/head.*")),
            )
        return nnx.Nothing

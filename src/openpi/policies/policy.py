from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import TorchAoConfig, AutoProcessor, PaliGemmaForConditionalGeneration
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.model = model
        #self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_actions = model.sample_actions
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        out_dict = self._sample_actions(
            sample_rng,
            _model.Observation.from_dict(inputs),
            **self._sample_kwargs,
        )
        outputs = {
            "state": inputs["state"],
            **out_dict,
        }
        # check if model is a Pi0 model and unbatch and convert to np.ndarray. 
        
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        # make a copy of the outputs to avoid modifying the original
        outputs_copy = jax.tree.map(lambda x: x.copy(), outputs)
        # apply output transformations
        outputs_copy = self._output_transform(outputs_copy)

        # replace the keys in the original outputs with the transformed keys
        outputs.update(outputs_copy)

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        vision_language_encoding = outputs['prefix_out']
        
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

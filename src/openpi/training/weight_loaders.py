import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0SmallFromPi0BaseWeightLoader(WeightLoader):
    """Loads the 300M action expert weights from a Pi0 checkpoint for Pi0-Small.

    Remaps the _1-suffixed action expert params to unsuffixed names (since Pi0-Small
    uses a single expert at index 0). Also transfers SigLIP encoder weights and
    projection layers.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")

        remapped = {}
        for k, v in flat_loaded.items():
            parts = k.rsplit("/", 1)
            if len(parts) == 2:
                prefix, leaf = parts
            else:
                prefix, leaf = "", parts[0]

            # Remap _1 suffixed params (action expert) to unsuffixed (single expert).
            if leaf.endswith("_1"):
                new_leaf = leaf[:-2]  # strip "_1"
                new_key = f"{prefix}/{new_leaf}" if prefix else new_leaf
                if new_key in flat_ref:
                    remapped[new_key] = v
            elif k in flat_ref:
                # Direct transfer for matching keys (SigLIP encoder, projection layers).
                # Skip the LLM expert 0 params (2.7B width) that don't match 300M width.
                if flat_ref[k].shape == v.shape:
                    remapped[k] = v

        # Merge: use remapped where available, fall back to randomly initialized ref params.
        result = {}
        for k, v in flat_ref.items():
            if k in remapped:
                rv = remapped[k]
                result[k] = rv.astype(v.dtype) if rv.dtype != v.dtype else rv
            else:
                result[k] = v

        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Pi0SmallFromPaliGemmaWeightLoader(WeightLoader):
    """Loads SigLIP encoder weights from the official PaliGemma checkpoint for Pi0-Small.

    Only the SigLIP encoder weights transfer (the LLM weights are 2B width and don't
    match the 300M model). Everything else stays randomly initialized.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}

        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")

        # Only transfer params that match in shape (SigLIP encoder, not the head or LLM).
        result = {}
        for k, v in flat_ref.items():
            if k in flat_loaded and flat_loaded[k].shape == v.shape:
                lv = flat_loaded[k]
                result[k] = lv.astype(v.dtype) if lv.dtype != v.dtype else lv
            else:
                result[k] = v

        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Pi0SmallScratchWeightLoader(WeightLoader):
    """Loads only SigLIP encoder weights for Pi0-Small, everything else from scratch.

    Uses the PaliGemma checkpoint for SigLIP encoder initialization only.
    """

    def load(self, params: at.Params) -> at.Params:
        # Delegate to the PaliGemma loader which does shape-matched transfer.
        return Pi0SmallFromPaliGemmaWeightLoader().load(params)


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

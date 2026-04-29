import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_forklift_example() -> dict:
    """Creates a random input example for the forklift policy."""
    return {
        "observation/state": np.random.rand(10).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Engage the forks under the pallet on the ground",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ForkliftInputs(transforms.DataTransformFn):
    """Map the forklift dataset's keys into the model's expected input dict.

    The forklift only has one camera (the top-mounted ZED2i warped left rect color image),
    mapped to ``base_0_rgb``. Both wrist slots are zero-padded and masked off.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # pi0 masks padding to False; pi0-FAST always uses True (matches LiberoInputs).
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ForkliftOutputs(transforms.DataTransformFn):
    """Trim model action output back to the 5 forklift axes (drop pi0 padding).

    Order: [DRIVE, STEER, LIFT, SHIFT, TILT]. Each value is the command for whichever
    control channel is active on that axis (vel for DRIVE, pos for the rest in stage_3).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :5])}

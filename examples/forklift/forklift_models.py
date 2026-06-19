"""Registry of fine-tuned forklift checkpoints + resolution to a local dir.

Single source of truth for "which checkpoints exist and how to load them", shared
by the serve wrapper (``serve.py``), the docker compose, and the docs. Add a new
checkpoint here once and it's selectable everywhere by its short name.

A "source" can be any of:
  * a short name registered in ``MODELS`` (e.g. ``full-10000``),
  * a Hugging Face repo id (e.g. ``tduggan93/pi05-forklift-full-10000``) — fetched
    with ``huggingface_hub.snapshot_download`` and cached,
  * a ``gs://`` path or a local checkpoint directory — passed straight through.

This indirection exists because openpi's ``download.maybe_download`` only knows how
to fetch ``gs://`` and local paths, *not* bare HF repo ids. Resolving here means
``--policy.dir=tduggan93/...`` style HF repos actually work.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """A selectable checkpoint: which train config builds the model, and where the
    weights live (HF repo id, gs:// path, or local dir)."""

    config: str
    source: str
    note: str = ""


# Short name -> spec. Keep this in sync with the table in README.md / INFERENCE.md.
MODELS: dict[str, ModelSpec] = {
    "lora": ModelSpec(
        config="pi05_forklift_lora",
        source="tduggan93/pi05-forklift-lora",
        note="LoRA finetune, fits ~24 GB VRAM.",
    ),
    "full": ModelSpec(
        config="pi05_forklift",
        source="tduggan93/pi05-forklift-full",
        note="Full finetune (final).",
    ),
    "full-10000": ModelSpec(
        config="pi05_forklift",
        source="tduggan93/pi05-forklift-full-10000",
        note="Full finetune @ 10k steps.",
    ),
    "full-15000": ModelSpec(
        config="pi05_forklift",
        source="tduggan93/pi05-forklift-full-15000",
        note="Full finetune @ 15k steps.",
    ),
}


def _looks_like_hf_repo(source: str) -> bool:
    """A bare 'owner/name' with no scheme and not an existing local path."""
    if "://" in source or source.startswith("gs:"):
        return False
    if pathlib.Path(source).exists():
        return False
    parts = source.strip("/").split("/")
    return len(parts) == 2 and all(parts)


def resolve_source(source: str) -> str:
    """Turn a source into something ``create_trained_policy`` can load directly.

    HF repo ids are snapshot-downloaded and the local cache dir is returned.
    Everything else (gs:// paths, local dirs) is passed through unchanged so that
    openpi's own ``maybe_download`` handles it.
    """
    if _looks_like_hf_repo(source):
        from huggingface_hub import snapshot_download

        logging.info("fetching HF checkpoint %s (cached after first download) ...", source)
        local = snapshot_download(repo_id=source)
        logging.info("  -> %s", local)
        return local
    return source


def resolve(model: str, *, config_override: str | None = None) -> tuple[str, str]:
    """Resolve a model name-or-source to ``(train_config_name, local_or_remote_dir)``.

    Args:
        model: a short name in ``MODELS``, or a raw source (HF repo / gs:// / local dir).
        config_override: required train config when ``model`` is a raw source that
            isn't in the registry (e.g. a local checkpoint you just trained).
    """
    if model in MODELS:
        spec = MODELS[model]
        return spec.config, resolve_source(spec.source)

    # Raw source not in the registry — need to be told which config built it.
    if config_override is None:
        known = ", ".join(sorted(MODELS))
        raise ValueError(
            f"{model!r} is not a known model name ({known}). If it's a raw checkpoint "
            f"path/repo, also pass the train config via --config (e.g. --config pi05_forklift)."
        )
    return config_override, resolve_source(model)


def describe() -> str:
    lines = ["Available forklift models:"]
    for name, spec in MODELS.items():
        lines.append(f"  {name:<12} config={spec.config:<20} {spec.source}  # {spec.note}")
    return "\n".join(lines)

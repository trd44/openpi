#!/usr/bin/env python3
"""Serve a fine-tuned forklift policy by short name.

Thin convenience wrapper around openpi's policy server: it resolves a friendly
model name (or an HF repo / local path) via ``forklift_models.py`` — downloading
from Hugging Face if needed — and starts the exact same websocket server that
``scripts/serve_policy.py`` does. The eval client and the ROS 2 inference node
talk to this server identically regardless of which checkpoint is loaded.

Examples:
    # by short name (downloads from HF on first use, then cached)
    uv run examples/forklift/serve.py --model full-10000
    uv run examples/forklift/serve.py --model full-15000
    uv run examples/forklift/serve.py --model lora

    # a local checkpoint you just trained (give the matching train config)
    uv run examples/forklift/serve.py \\
        --model checkpoints/pi05_forklift_lora/forklift_lora_v1/29999 \\
        --config pi05_forklift_lora

    # list what's registered
    uv run examples/forklift/serve.py --list
"""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

try:  # works whether run as a script (`uv run examples/forklift/serve.py`) or imported
    from examples.forklift import forklift_models
except ModuleNotFoundError:
    import forklift_models


@dataclasses.dataclass
class Args:
    model: str = "full-15000"
    """Short name (lora, full, full-10000, full-15000) or a raw HF repo / gs:// /
    local checkpoint dir. See --list."""
    config: str | None = None
    """Train config name. Only needed when --model is a raw path/repo not in the
    registry (e.g. a local checkpoint)."""
    port: int = 8000
    """Websocket port to serve on."""
    default_prompt: str | None = None
    """Prompt injected when an observation doesn't carry one."""
    list: bool = False
    """Print the model registry and exit."""


def main(args: Args) -> None:
    if args.list:
        print(forklift_models.describe())
        return

    config_name, checkpoint_dir = forklift_models.resolve(args.model, config_override=args.config)
    logging.info("serving model %r: config=%s dir=%s", args.model, config_name, checkpoint_dir)

    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name), checkpoint_dir, default_prompt=args.default_prompt
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s) on port %d", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"model": args.model, "config": config_name, "checkpoint": str(checkpoint_dir)},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

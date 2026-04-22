"""Utilities for loading HiFi-GAN UNIVERSAL_V1 checkpoint."""

import json
from pathlib import Path

from .generator import HiFiGANGenerator


def load_hifigan_generator(checkpoint_path: str, device: str = "cpu") -> HiFiGANGenerator:
    """Load HiFi-GAN generator from a jik876/hifi-gan checkpoint directory.

    Expects the directory to contain `generator_v1` (state dict) and `config.json`.
    """
    import torch

    ckpt_dir = Path(checkpoint_path)
    config_path = ckpt_dir / "config.json"
    gen_path = ckpt_dir / "generator_v1"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        assert config["num_mels"] == 80, f"Expected 80 mels, got {config['num_mels']}"
    else:
        config = None

    generator = HiFiGANGenerator()
    state_dict = torch.load(gen_path, map_location=device, weights_only=False)
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator.to(device)

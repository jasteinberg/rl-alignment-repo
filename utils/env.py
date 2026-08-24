"""
utils/env.py
============
Single source of truth for all path and device configuration.
Import this in every notebook instead of hardcoding paths.

Usage:
    import sys; sys.path.insert(0, ENV.REPO)
    from utils.env import ENV
"""

import os
import torch
from dataclasses import dataclass

# Repo root resolved from this file's location, so paths are correct
# wherever the repo is checked out (no hardcoded user paths).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# HF model cache root; override with HF_MODELS_ROOT (default ~/models).
_MODELS_ROOT = os.environ.get("HF_MODELS_ROOT", os.path.expanduser("~/models"))
# Third-party datasets checked out outside the repo; override with
# EXTERNAL_ROOT (default ~/external).
_EXTERNAL_ROOT = os.environ.get("EXTERNAL_ROOT", os.path.expanduser("~/external"))


@dataclass(frozen=True)
class _Env:
    # Paths
    ROOT      : str = os.path.dirname(_REPO_ROOT)
    REPO      : str = _REPO_ROOT
    MODELS    : str = os.path.join(_MODELS_ROOT, "transformers")
    HUB       : str = os.path.join(_MODELS_ROOT, "hub")
    DATA      : str = os.path.join(_REPO_ROOT, "data")
    EXTERNAL  : str = _EXTERNAL_ROOT
    GOT       : str = os.path.join(_EXTERNAL_ROOT, "geometry-of-truth", "datasets")
    NOTEBOOKS : str = os.path.join(_REPO_ROOT, "notebooks")
    ARTIFACTS : str = os.path.join(_REPO_ROOT, "artifacts")

    # Device
    DEVICE    : str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # HuggingFace cache root (override with HF_MODELS_ROOT)
    HF_CACHE  : str = os.path.join(_MODELS_ROOT, "hub")

    def summary(self) -> None:
        print(f"Device:    {self.DEVICE}")
        print(f"Root:      {self.ROOT}")
        print(f"Repo:      {self.REPO}")
        print(f"Models:    {self.MODELS}")
        print(f"Data:      {self.DATA}")
        print(f"External:  {self.EXTERNAL}")
        print(f"HF cache:  {self.HF_CACHE}")


ENV = _Env()

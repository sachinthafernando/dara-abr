"""DARA-ABR: counterfactual reward diagnosis for learning-based adaptive bitrate streaming on 5G mmWave.

The environment, training loop and diagnosis run in notebooks/dara_abr.ipynb (one run = one notebook execution).
This package holds what the notebook and the scripts share: the mmWave network generator, the experiment
registry (training environments, methods, test conditions) and the statistics used in the paper.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

"""
Wavespeed AI Python Client

A Python client for interacting with the Wavespeed AI API.
"""

__version__ = "0.1.0"

from .client import WaveSpeed
from .schemas.prediction import Prediction, PredictionUrls, PredictionResponse

__all__ = [
    "WaveSpeed",
    "Prediction",
    "PredictionUrls",
    "PredictionResponse",
    "__version__",
]
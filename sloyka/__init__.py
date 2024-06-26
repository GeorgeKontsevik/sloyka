import sys

from loguru import logger

from .src import (
    AreaMatcher,
    City_services,
    EmotionRecognizer,
    EventDetection,
    Geocoder,
    GeoDataGetter,
    Semgraph,
    TextClassifiers,
    VKParser,
)

__all__ = [
    "EventDetection",
    "TextClassifiers",
    "Geocoder",
    "GeoDataGetter",
    "Semgraph",
    "VKParser",
    "City_services",
    "AreaMatcher",
    "EmotionRecognizer",
]

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:MM-DD HH:mm}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)

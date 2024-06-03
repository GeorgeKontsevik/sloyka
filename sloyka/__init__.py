from loguru import logger
import sys

from .src import (
    EventDetection,
    Geocoder,
    TextClassifiers,
    GeoDataGetter,
    Semgraph,
    Streets, 
    NER_parklike,
    VKParser,
<<<<<<< HEAD
    Pipeline
=======
    City_services,
    AreaMatcher
>>>>>>> origin/feat/group_names
)

__all__ = [
    "EventDetection",
    "TextClassifiers",
    "Geocoder",
    "GeoDataGetter",
    "Semgraph",
    "Streets", 
    "NER_parklike",
    "VKParser",
<<<<<<< HEAD
    "Pipeline"
=======
    "City_services",
    "AreaMatcher"
>>>>>>> origin/feat/group_names
]

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:MM-DD HH:mm}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)

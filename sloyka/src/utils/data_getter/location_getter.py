import logging
import warnings
from typing import List, Optional

from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GeocodingError(Exception):
    """Custom exception for geocoding errors"""

    pass


class Location:
    """
    This class is aimed to efficiently geocode addresses using Nominatim.
    Geocoded addresses are stored in the 'book' dictionary argument.
    Thus, if the address repeats -- it would be taken from the book.
    """

    max_tries = 3
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.geolocator = Nominatim(user_agent="soika")
        self.addr = []
        self.book = {}

    def geocode_with_retry(self, query: str) -> Optional[List[float]]:
        """
        Function to handle 403 error while geocoding using Nominatim.
        TODO: 1. Provide an option to use alternative geocoder
        TODO: 2. Wrap this function as a decorator
        """
        self.logger.info(f"Geocoding query: {query}")
        for _ in range(Location.max_tries):
            try:
                geocode = self.geolocator.geocode(
                    query, addressdetails=True, language="ru"
                )
                self.logger.debug(f"Geocode result: {geocode}")
                return geocode
            except GeocoderUnavailable as e:
                self.logger.warning(
                    f"Geocoder unavailable, retrying ({_+1}/{Location.max_tries}): {e}"
                )
                continue
        self.logger.error(
            f"Failed to geocode after {Location.max_tries} tries: {query}"
        )
        raise GeocodingError(
            f"Failed to geocode after {Location.max_tries} tries: {query}"
        )

    def query(self, address: str) -> Optional[List[float]]:
        """
        A function to query the address and return its geocode if available.

        :param address: A string representing the address to be queried.
        :return: An optional list of floats representing the geocode of the address, or None if not found.
        """
        self.logger.info(f"Querying address: {address}")
        if address not in self.book:
            query = f"{address}"
            try:
                res = self.geocode_with_retry(query)
                self.book[address] = res
            except GeocodingError as e:
                self.logger.error(f"Geocoding error: {e}")
                return None
        return self.book.get(address)

# addr_extractor.py
import warnings

from natasha.extractors import Extractor, Match

from ..utils.data_processing.rule_for_natasha import ADDR_PART

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class AddrExtractorError(Exception):
    """Custom exception for address extractor errors"""

    pass


class AddressExtractorExtra(Extractor):
    """
    Extractor for addresses
    """

    # logger = logging.getLogger(__name__)

    def __init__(self, morph):
        """
        Initialize the address extractor

        :param morph: Morphological analyzer
        """
        super().__init__(ADDR_PART, morph)

    def find(self, text):
        """
        Extract addresses from the given text

        :param text: Input text
        :return: Match object containing the extracted address
        """
        # self.logger.info(f"Extracting addresses from text: {text}")
        matches = self(text)
        if not matches:
            # self.logger.debug("No matches found")
            return

        matches = sorted(matches, key=lambda _: _.start)
        if not matches:
            # self.logger.debug("No matches found after sorting")
            return

        start = matches[0].start
        stop = matches[-1].stop
        parts = [_.fact for _ in matches]
        # self.logger.debug(f"Extracted address parts: {parts}")
        try:
            # FIXME: Undefined name `obj`
            return Match(start, stop, obj.Addr(parts))
        except Exception as e:
            # self.logger.error(f"Error creating Match object: {e}")
            raise AddrExtractorError(f"Error creating Match object: {e}")

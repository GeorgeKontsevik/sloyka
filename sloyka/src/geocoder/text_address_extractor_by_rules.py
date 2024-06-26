import re
from typing import Optional

from loguru import logger
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)

from sloyka.src.utils.constants import EXCEPTIONS_CITY_COUNTRY


class NatashaExtractor:
    def __init__(self):
        NatashaExtractor.segmenter = Segmenter()
        NatashaExtractor.morph_vocab = MorphVocab()
        NatashaExtractor.emb = NewsEmbedding()
        NatashaExtractor.morph_tagger = NewsMorphTagger(NatashaExtractor.emb)
        NatashaExtractor.syntax_parser = NewsSyntaxParser(NatashaExtractor.emb)
        NatashaExtractor.ner_tagger = NewsNERTagger(NatashaExtractor.emb)

    @staticmethod
    def get_ner_address_natasha(text) -> Optional[str]:
        """
        Extract street names from text using the Natasha library if BERT could not.

        This function uses the Natasha library to extract street names from the text column
        when BERT has not provided a result. It handles text cleaning, NER tagging, and filtering
        based on predefined exceptions.

        Args:
            row (pd.Series): The row of the DataFrame containing the text and street information.
            text_col (str): The name of the column in the DataFrame with the text to process.

        Returns:
            Optional[str]: The extracted street name if found, otherwise None.
        """
        try:
            # Check if 'Street' is already extracted
            if isinstance(text, str):
                # Clean and prepare text
                cleaned_text = NatashaExtractor._clean_text(text)
                doc = NatashaExtractor._process_text_with_natasha(cleaned_text)

                # Extract and filter locations
                location_spans = NatashaExtractor._extract_location_spans(doc)
                filtered_locations = NatashaExtractor._filter_locations(location_spans)

                # Return the first valid location
                return filtered_locations[0] if filtered_locations else None
            else:
                return text
        except Exception as e:
            logger.warning(f"Error in get_ner_address_natasha with {text}]: {e}")
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean the input text by removing unwanted patterns.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        return re.sub(r"\[.*?\]", "", text)

    @staticmethod
    def _process_text_with_natasha(text: str) -> Doc:
        """
        Process text with the Natasha library to perform segmentation, tagging, and NER.

        Args:
            text (str): The cleaned input text.

        Returns:
            Doc: The processed Natasha Doc object.
        """
        try:
            doc = Doc(text)
            doc.segment(NatashaExtractor.segmenter)
            doc.tag_morph(NatashaExtractor.morph_tagger)
            doc.parse_syntax(NatashaExtractor.syntax_parser)
            doc.tag_ner(NatashaExtractor.ner_tagger)
            for span in doc.spans:
                span.normalize(NatashaExtractor.morph_vocab)
            return doc
        except Exception as e:
            logger.warning(f"Error processing text with Natasha: {e}")
            return Doc("")

    @staticmethod
    def _extract_location_spans(doc: Doc) -> list:
        """
        Extract location spans from the processed Natasha Doc object.

        Args:
            doc (Doc): The processed Natasha Doc object.

        Returns:
            list: List of location spans.
        """
        try:
            return [span for span in doc.spans if span.type == "LOC"]
        except Exception as e:
            logger.warning(f"Error extracting location spans: {e}")
            return []

    @staticmethod
    def _filter_locations(spans: list) -> list:
        """
        Filter out locations based on predefined exceptions.

        Args:
            spans (list): List of location spans.

        Returns:
            list: Filtered list of location texts.
        """
        try:
            return [
                span.text
                for span in spans
                if span.normal.lower() not in map(str.lower, EXCEPTIONS_CITY_COUNTRY)
            ]
        except Exception as e:
            logger.warning(f"Error filtering locations: {e}")
            return []

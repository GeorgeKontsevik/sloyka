import math
import re
import string
from typing import List, Optional, Tuple

import pandas as pd

# Initialize morphological analyzer (use the correct library for your context)
import pymorphy2
from flair.data import Sentence
from loguru import logger
from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()

from sloyka.src.geocoder.text_address_extractor_by_rules import NatashaExtractor
from sloyka.src.utils.constants import (
    END_INDEX_POSITION,
    REPLACEMENT_DICT,
    SCORE_THRESHOLD,
    START_INDEX_POSITION,
    TARGET_TOPONYMS,
)


class StreetExtractor:
    extractor = NatashaExtractor()

    @staticmethod
    def process_pipeline(
        df: pd.DataFrame, text_column: str, classifier
    ) -> pd.DataFrame:
        local_df = df.copy()
        """
        Execute the address extraction pipeline on the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            text_column (str): Column name in the DataFrame with text data for address extraction.

        Returns:
            pd.DataFrame: DataFrame with extracted street addresses and additional processing columns.
        """
        texts = StreetExtractor._preprocess_text_column(local_df, text_column)
        extracted_streets = StreetExtractor._extract_streets(texts, classifier)
        refined_streets = StreetExtractor._refine_street_data(extracted_streets)
        building_numbers = StreetExtractor._extract_building_numbers(
            texts, refined_streets
        )
        toponyms = StreetExtractor._extract_toponyms(texts, refined_streets)

        # Combine results into a DataFrame
        processed_df = pd.DataFrame(
            {
                text_column: texts,
                "Street": refined_streets,
                "Numbers": building_numbers,
                "Toponyms": toponyms,
            }
        )

        StreetExtractor._check_df_len_didnt_change(local_df, processed_df)

        return processed_df

    @staticmethod
    def _check_df_len_didnt_change(df1, df2):
        try:
            assert len(df1) == len(df2)
        except Exception as e:
            logger.critical("dfs lengths differ")
            raise e

    @staticmethod
    def _preprocess_text_column(df: pd.DataFrame, text_column: str) -> List[str]:
        """
        Preprocess the text column by ensuring non-null values and converting to string type.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            text_column (str): Column name in the DataFrame with text data.

        Returns:
            List[str]: List of preprocessed text entries.
        """
        try:
            text_series = df[text_column].dropna().astype(str)
            return text_series.tolist()
        except Exception as e:
            logger.warning(f"Error in _preprocess_text_column: {e}")
            return []

    @staticmethod
    def _extract_streets(
        texts: List[str], classifier
    ) -> List[Tuple[Optional[str], Optional[float]]]:
        """
        Extract street names from the text column using NER model.

        Args:
            texts (List[str]): List of text entries.

        Returns:
            List[Tuple[Optional[str], Optional[float]]]: List of tuples with extracted street names and confidence scores.
        """
        tqdm.pandas()
        extracted_streets = []
        for text in tqdm(texts):
            try:
                extracted_streets.append(
                    StreetExtractor.extract_ner_street(text, classifier)
                )
            except Exception as e:
                logger.warning(f"Error extracting NER street from text '{text}': {e}")
                extracted_streets.append((None, None))
        return extracted_streets

    @staticmethod
    def _refine_street_data(
        street_data: List[Tuple[Optional[str], Optional[float]]],
    ) -> List[Optional[str]]:
        """
        Refine street data by normalizing and cleaning up street names.

        Args:
            street_data (List[Tuple[Optional[str], Optional[float]]]): List of tuples with extracted street names and confidence scores.

        Returns:
            List[Optional[str]]: List of refined street names.
        """
        refined_streets = []
        for street, _ in street_data:
            if street:
                try:
                    refined_streets.append(StreetExtractor._refine_street_name(street))
                except Exception as e:
                    logger.warning(f"Error refining street '{street}': {e}")
                    refined_streets.append(None)
            else:
                refined_streets.append(None)
        return refined_streets

    @staticmethod
    def _refine_street_name(street: str) -> str:
        """
        Refine street name by normalizing and cleaning up the street string.

        Args:
            street (str): Raw street name.

        Returns:
            str: Refined street name.
        """
        try:
            street = re.sub(r"(\D)(\d)(\D)", r"\1 \2\3", street)
            street = re.sub(r"\d+", "", street).strip().lower()
            return street
        except Exception as e:
            logger.warning(f"Error in _refine_street_name with street '{street}': {e}")
            return ""

    @staticmethod
    def _extract_building_numbers(
        texts: List[str], streets: List[Optional[str]]
    ) -> List[Optional[str]]:
        """
        Extract building numbers from the text data.

        Args:
            texts (List[str]): List of text entries.
            streets (List[Optional[str]]): List of refined street names.

        Returns:
            List[Optional[str]]: List of extracted building numbers.
        """
        building_numbers = []
        for text, street in zip(texts, streets):
            if street:
                try:
                    building_numbers.append(
                        StreetExtractor._extract_building_number(text, street)
                    )
                except Exception as e:
                    logger.warning(
                        f"Error extracting building number from text '{text}' with street '{street}': {e}"
                    )
                    building_numbers.append(None)
            else:
                building_numbers.append(None)
        return building_numbers

    @staticmethod
    def _extract_building_number(text: str, street: str) -> str:
        """
        Extract building number from the text.

        Args:
            text (str): Input text for address extraction.
            street (str): Extracted and refined street name.

        Returns:
            str: Extracted building number.
        """
        try:
            numbers = " ".join(re.findall(r"\d+", text))
            return StreetExtractor.extract_building_num(text, street, numbers)
        except Exception as e:
            logger.warning(
                f"Error in _extract_building_number with text '{text}' and street '{street}': {e}"
            )
            return ""

    @staticmethod
    def _extract_toponyms(
        texts: List[str], streets: List[Optional[str]]
    ) -> List[Optional[str]]:
        """
        Extract toponyms from the text data.

        Args:
            texts (List[str]): List of text entries.
            streets (List[Optional[str]]): List of refined street names.

        Returns:
            List[Optional[str]]: List of extracted toponyms.
        """
        toponyms = []
        for text, street in zip(texts, streets):
            if street:
                try:
                    toponyms.append(StreetExtractor.extract_toponym(text, street))
                except Exception as e:
                    logger.warning(
                        f"Error extracting toponym from text '{text}' with street '{street}': {e}"
                    )
                    toponyms.append(None)
            else:
                toponyms.append(None)
        return toponyms

    @staticmethod
    def extract_toponym(text: str, street_name: str) -> Optional[str]:
        """
        Extract toponyms near the specified street name in the text.

        This function identifies the position of a street name in the text and searches for related toponyms
        within a specified range around the street name.

        Args:
            text (str): The text containing the address.
            street_name (str): The name of the street to search around.

        Returns:
            Optional[str]: The first toponym found if present, otherwise None.
        """
        try:
            # Handle the case where text is NaN
            if isinstance(text, float) and math.isnan(text):
                return None

            # Clean and split the text into words
            cleaned_text = StreetExtractor._clean_text(text)
            words = cleaned_text.split()

            # Find positions of the street name
            positions = StreetExtractor._find_street_name_positions(words, street_name)
            if not positions:
                return None

            # Search for toponyms in the range around the street name
            toponym = StreetExtractor._search_toponyms(words, positions[0])
            return toponym

        except Exception as e:
            logger.warning(
                f"Error in extract_toponym with text '{text}' and street_name '{street_name}': {e}"
            )
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean the input text by removing punctuation and converting to lowercase.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        return text.translate(str.maketrans("", "", string.punctuation)).lower()

    @staticmethod
    def _find_street_name_positions(words: List[str], street_name: str) -> List[int]:
        """
        Find positions of the street name in the list of words.

        Args:
            words (List[str]): List of words from the cleaned text.
            street_name (str): The name of the street to find.

        Returns:
            List[int]: List of positions where the street name occurs.
        """
        return [index for index, word in enumerate(words) if word == street_name]

    @staticmethod
    def _search_toponyms(words: List[str], position: int) -> Optional[str]:
        """
        Search for toponyms within a specified range around the given position.

        Args:
            words (List[str]): List of words from the cleaned text.
            position (int): The position around which to search for toponyms.

        Returns:
            Optional[str]: The first toponym found if present, otherwise None.
        """
        search_start = max(0, position - START_INDEX_POSITION)
        search_end = min(len(words), position + END_INDEX_POSITION)

        for i in range(search_start, search_end + 1):
            word = words[i]
            try:
                normal_form = morph.parse(word)[0].normal_form
            except Exception as e:
                logger.warning(f"Error parsing word '{word}': {e}")
                continue

            if normal_form in TARGET_TOPONYMS:
                return REPLACEMENT_DICT.get(normal_form, normal_form)

        return None

    @staticmethod
    def extract_building_num(text: str, street_name: str, number: Optional[str]) -> str:
        """
        Extract building numbers near the specified street name in the text.

        This function identifies the position of a street name in the text and searches for related building numbers
        within a specified range of indexes around the street name.

        Args:
            text (str): The text containing the address.
            street_name (str): The name of the street to search around.
            number (Optional[str]): Previously extracted building number.

        Returns:
            str: The first building number found if present, otherwise an empty string.
        """
        try:
            if isinstance(text, float) and math.isnan(text):
                return ""

            cleaned_text = StreetExtractor._clean_text(text)
            words = cleaned_text.split()

            positions = StreetExtractor._find_street_name_positions(words, street_name)
            if not positions:
                return ""

            building_number = StreetExtractor._search_building_number(
                words, positions[0]
            )
            return building_number

        except Exception as e:
            logger.warning(
                f"Error in extract_building_num with text '{text}', street_name '{street_name}', number '{number}': {e}"
            )
            return ""

    @staticmethod
    def _find_street_name_positions(words: List[str], street_name: str) -> List[int]:
        """
        Find positions of the street name in the list of words.

        Args:
            words (List[str]): List of words from the cleaned text.
            street_name (str): The name of the street to find.

        Returns:
            List[int]: List of positions where the street name occurs.
        """
        return [index for index, word in enumerate(words) if word == street_name]

    @staticmethod
    def _search_building_number(words: List[str], position: int) -> str:
        """
        Search for building numbers within a specified range around the given position.

        Args:
            words (List[str]): List of words from the cleaned text.
            position (int): The position around which to search for building numbers.

        Returns:
            str: The first building number found if present, otherwise an empty string.
        """
        search_start = max(0, position)
        search_end = min(len(words), position + END_INDEX_POSITION)

        for index in range(search_start, search_end):
            word = words[index]
            if StreetExtractor._is_building_number(word):
                return word

        return ""

    @staticmethod
    def _is_building_number(word: str) -> bool:
        """
        Check if a word is a valid building number.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word is a valid building number, otherwise False.
        """
        return any(character.isdigit() for character in word) and len(word) <= 3

    # ---------
    @staticmethod
    def extract_ner_street(text: str, classifier) -> pd.Series:
        """
        Extract street addresses from text using a pre-trained custom NER model.

        This function processes text by removing unnecessary content, applies a custom NER model
        to extract mentioned addresses, and returns the address with a confidence score.

        Args:
            text (str): The input text to process and extract addresses from.

        Returns:
            pd.Series: A Series containing the extracted address and confidence score,
                    or [None, None] if extraction fails or the score is below the threshold.
        """
        try:
            cleaned_text = StreetExtractor._clean_text(text)
            sentence = Sentence(cleaned_text)

            # Predict entities using the classifier
            classifier.predict(sentence)

            address, score = StreetExtractor._extract_address_and_score(sentence)

            if not address or score < SCORE_THRESHOLD:
                address = StreetExtractor.extractor.get_ner_address_natasha(text)
                if address:
                    score = 1

            # Return the result if the score is above the threshold
            return pd.Series(
                [address, score]
                if score is not None and score > SCORE_THRESHOLD
                else [None, None]
            )

        except Exception as e:
            logger.warning(f"Error in extract_ner_street with text '{text}': {e}")
            return pd.Series([None, None])

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean the input text by removing unwanted patterns.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        try:
            return re.sub(r"\[.*?\]", "", text)
        except Exception as e:
            logger.warning(f"Error in _clean_text with text '{text}': {e}")
            return text

    @staticmethod
    def _extract_address_and_score(
        sentence: Sentence,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract address and score from the NER model's predictions.

        Args:
            sentence (Sentence): The Sentence object containing NER predictions.

        Returns:
            Tuple[Optional[str], Optional[float]]: Extracted address and its confidence score.
        """
        try:
            labels = sentence.get_labels("ner")
            if labels:
                label = labels[0]
                address = StreetExtractor._parse_address(label.labeled_identifier)
                score = round(label.score, 3)
                return address, score
            return None, None
        except IndexError as e:
            logger.warning(f"Error in _extract_address_and_score: {e}")
            return None, None

    @staticmethod
    def _parse_address(label_value: str) -> str:
        """
        Parse the address from the label value string.

        Args:
            label_value (str): The labeled identifier from the NER model.

        Returns:
            str: Parsed address.
        """
        try:
            return label_value.split("]: ")[1].split("/")[0].replace('"', "")
        except IndexError as e:
            logger.warning(
                f"Error in _parse_address with label value '{label_value}': {e}"
            )
            return ""

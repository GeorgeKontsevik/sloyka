"""
TODO: add spellchecker since there might be misspelled words.

This module is aimed to provide necessary tools to find mentioned
location in the text. 

@class:Location:
A class aimed to efficiently geocode addresses using Nominatim. Geocoded addresses are stored in the 'book' dictionary argument. 
Thus, if the address repeats, it would be taken from the book.

@class:Streets: 
A class encapsulating functionality for retrieving street data
for a specified city from OSM and processing it to extract useful information for geocoding purposes.

@class:Geocoder:
A class providing functionality for simple geocoding and address extraction.
"""
import numpy as np
import re
import warnings
import os
import flair
import geopandas as gpd
import pandas as pd
import pymorphy2
import torch
import string
import math
from rapidfuzz import fuzz
from nltk.stem.snowball import SnowballStemmer
from sloyka.src.utils.data_getter.historical_geo_data_getter import HistGeoDataGetter
from sloyka.src.utils.constants import (
    AREA_STOPWORDS,
    GROUP_STOPWORDS,
    REGEX_PATTERN,
    REPLACEMENT_STRING,
)

from flair.models import SequenceTagger
from shapely.geometry import Point
from tqdm import tqdm
# from natasha import (
#     Segmenter,
#     MorphVocab,
#     NewsEmbedding,
#     NewsMorphTagger,
#     NewsSyntaxParser,
#     NewsNERTagger,
#     Doc,
# )

from loguru import logger

from pandarallel import pandarallel
from sloyka.src.geocoder.city_objects_extractor import OtherGeoObjects
from sloyka.src.utils.data_getter.street_getter import Streets
from sloyka.src.utils.data_getter.location_getter import Location
from sloyka.src.utils.data_getter.geo_data_getter import GeoDataGetter
from sloyka.src.utils.data_preprocessing.preprocessor import PreprocessorInput
from sloyka.src.geocoder.street_extractor import StreetExtractor
from sloyka.src.geocoder.word_form_matcher import WordFormFinder

pandarallel.initialize(progress_bar=True, nb_workers=-1)

# segmenter = Segmenter()
# morph_vocab = MorphVocab()
morph = pymorphy2.MorphAnalyzer()

# emb = NewsEmbedding()
# morph_tagger = NewsMorphTagger(emb)
# syntax_parser = NewsSyntaxParser(emb)
# ner_tagger = NewsNERTagger(emb)

stemmer = SnowballStemmer("russian")

tqdm.pandas()


class Geocoder:
    """
    This class provides a functionality of simple geocoder
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    global_crs: int = 4326

    def __init__(
        self,
        df,
        model_path: str = "Geor111y/flair-ner-addresses-extractor",
        device: str = "cpu",
        osm_id: int = None,
        city_tags: dict ={"place": ["state"]}
    ):
        self.df = PreprocessorInput().run(df)
        self.device = device
        flair.device = torch.device(device)
        self.classifier = SequenceTagger.load(model_path)
        self.osm_id = osm_id
        self.osm_city_name = (
            GeoDataGetter()
            .get_features_from_id(osm_id=self.osm_id,tags=city_tags, selected_columns=["name", "geometry"])
            .iloc[0]["name"]
        )
        self.street_names = Streets.run(self.osm_id)


    @staticmethod
    def get_stem(street_names_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function finds the stem of the word to find this stem in the street
        names dictionary (df).
        """
        logger.info("get_stem started")

        morph = pymorphy2.MorphAnalyzer()
        cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

        for case in cases:
            street_names_df[case] = street_names_df["street_name"].apply(
                lambda x: morph.parse(x)[0].inflect({case}).word if morph.parse(x)[0].inflect({case}) else None
            )
        return street_names_df


    @staticmethod
    def get_level(row: pd.Series) -> str:
        """
        Addresses in the messages are recognized on different scales:
        1. Where we know the street name and house number -- house level;
        2. Where we know only street name -- street level (with the centroid
        geometry of the street);
        3. Where we don't know any info but the city -- global level.
        """

        if (not pd.isna(row["Street"])) and (row["Numbers"] == ""):
            return "street"
        elif (not pd.isna(row["Street"])) and (row["Numbers"] != ""):
            return "house"
        else:
            return "global"

 
    def create_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Function simply creates gdf from the recognised geocoded geometries.
        """
        logger.info("create_gdf started")

        df["Location"] = df["addr_to_geocode"].progress_apply(Location().query)
        df = df.dropna(subset=["Location"])
        df["geometry"] = df['Location'].apply(lambda x: Point(x.longitude, x.latitude))
        df["Location"] = df['Location'].apply(lambda x: x.address)
        df["Numbers"].astype(str)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=Geocoder.global_crs)

        return gdf

    def set_global_repr_point(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        This function set the centroid (actually, representative point) of the
        geocoded addresses to those texts that weren't geocoded (or didn't
        contain any addresses according to the trained NER model).
        """

        try:
            gdf.loc[gdf["level"] == "global", "geometry"] = gdf.loc[
                gdf["level"] != "global", "geometry"
            ].unary_union.representative_point()
        except AttributeError:
            pass

        return gdf

    def merge_to_initial_df(self, gdf: gpd.GeoDataFrame, initial_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        This function merges geocoded df to the initial df in order to keep
        all original attributes.
        """

        # initial_df.drop(columns=['key_0'], inplace=True)
        gdf = initial_df.join(
            gdf[
                [
                    "Street",
                    "initial_street",
                    "only_full_street_name",
                    "Numbers",
                    "Score",
                    "location_options",
                    "Location",
                    "geometry",
                ]
            ],
            how="outer",
        )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=Geocoder.global_crs)

        return

    def assign_street(variable):
        if isinstance(variable, float) and math.isnan(variable):
            return "street"
        return variable

    def get_df_areas(self, osm_id, tags):
        """
        Retrieves the GeoDataFrame of areas corresponding to the given OSM ID and tags.

        Args:
            osm_id (int): The OpenStreetMap ID.
            tags (dict): The tags to filter by.
            date (str): The date of the data to retrieve.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame containing the areas.

        This function first checks if the GeoDataFrame corresponding to the given OSM ID is already in the cache.
        If it is, it returns the cached GeoDataFrame. Otherwise, it retrieves the GeoDataFrame from the HistGeoDataGetter,
        filters out the 'way' elements, and adds it to the cache. Finally, it returns the GeoDataFrame from the cache.
        """
        area_cache = {}
        if osm_id not in area_cache:
            geo_data_getter = HistGeoDataGetter()
            df_areas = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags)
            df_areas = df_areas[df_areas["element_type"] != "way"]
            area_cache[osm_id] = df_areas
        return area_cache[osm_id]

    def preprocess_group_name(self, group_name):
        """
        Preprocesses a group name by converting it to lowercase, removing special characters, and removing specified stopwords.

        Args:
            group_name (str): The group name to preprocess.

        Returns:
            str: The preprocessed group name.
        """
        group_name = group_name.lower()
        group_name = re.sub(REGEX_PATTERN, REPLACEMENT_STRING, group_name)
        words_to_remove = GROUP_STOPWORDS
        for word in words_to_remove:
            group_name = re.sub(word, "", group_name, flags=re.IGNORECASE)
        return group_name

    def preprocess_area_names(self, df_areas):
        """
        Preprocesses the area names in the given DataFrame by removing specified stopwords, converting the names to lowercase,
        and stemming them.

        Parameters:
            df_areas (DataFrame): The DataFrame containing the area names.

        Returns:
            DataFrame: The DataFrame with preprocessed area names, where the 'area_name' column contains the original names
                      with stopwords removed, the 'area_name_processed' column contains the lowercase names with special characters
                      removed, and the 'area_stems' column contains the stemmed names.
        """
        words_to_remove = AREA_STOPWORDS
        for word in words_to_remove:
            df_areas["area_name"] = df_areas["name"].str.replace(word, "", regex=True)

        df_areas["area_name_processed"] = df_areas["area_name"].str.lower()
        df_areas["area_name_processed"] = df_areas["area_name_processed"].str.replace(
            REGEX_PATTERN, REPLACEMENT_STRING, regex=True
        )
        df_areas["area_stems"] = df_areas["area_name_processed"].apply(
            lambda x: [stemmer.stem(word) for word in x.split()]
        )
        return df_areas

    def match_group_to_area(self, group_name, df_areas):
        """
        Matches a given group name to an area in a DataFrame of areas.

        Args:
            group_name (str): The name of the group to match.
            df_areas (DataFrame): The DataFrame containing the areas to match against.

        Returns:
            tuple: A tuple containing the best match for the group name and the admin level of the match.
                   If no match is found, returns (None, None).
        """
        group_name_stems = [stemmer.stem(word) for word in group_name.split()]
        max_partial_ratio = 20
        max_token_sort_ratio = 20
        best_match = None
        admin_level = None

        for _, row in df_areas.iterrows():
            area_stems = row["area_stems"]

            partial_ratio = fuzz.partial_ratio(group_name, row["area_name_processed"])
            token_sort_ratio = fuzz.token_sort_ratio(group_name_stems, area_stems)

            if partial_ratio > max_partial_ratio and token_sort_ratio > max_token_sort_ratio:
                max_partial_ratio = partial_ratio
                max_token_sort_ratio = token_sort_ratio
                best_match = row["area_name"]
                admin_level = row["key"]

        return best_match, admin_level

    def run(
        self, df: pd.DataFrame = None, tags:dict|None=None, text_column: str = "text", group_column: str | None = "group_name", search_for_objects=False
    ):
        """
        Runs the data processing pipeline on the input DataFrame.

        Args:
            tags (dict): The tags to filter by.
            date (str): The date of the data to retrieve.
            df (pd.DataFrame): The input DataFrame.
            text_column (str, optional): The name of the text column in the DataFrame. Defaults to "text".

        Returns:
            gpd.GeoDataFrame: The processed DataFrame after running the data processing pipeline.

        This function retrieves the GeoDataFrame of areas corresponding to the given OSM ID and tags.
        It then preprocesses the area names and matches each group name to an area. The best match
        and admin level are assigned to the DataFrame. The function also retrieves other geographic
        objects and street names, preprocesses the street names, finds the word form, creates a GeoDataFrame,
        merges it with the other geographic objects, assigns the street tag, and returns the final GeoDataFrame.
        """
        df = self.df
        initial_df = df.copy()

        if search_for_objects:
            df_obj = OtherGeoObjects.run(self.osm_id, df, text_column)
            

        if tags:
            df_areas = self.get_df_areas(self.osm_id, tags)
            df_areas = self.preprocess_area_names(df_areas)

            if group_column and group_column in df.columns:
                for i, group_name in enumerate(df[group_column]):
                    processed_group_name = self.preprocess_group_name(group_name)
                    best_match, admin_level = self.match_group_to_area(processed_group_name, df_areas)
                    df.at[i, "territory"] = best_match
                    df.at[i, "key"] = admin_level
            del df_areas
        # df = AreaMatcher.run(self, df, osm_id, tags, date)

        df[text_column] = df[text_column].astype(str).str.replace('\n', ' ')
        df[text_column] = df[text_column].apply(str)
        

        df = StreetExtractor.process_pipeline(df, text_column, self.classifier)
        street_names = self.get_stem(self.street_names)

        df = WordFormFinder(self.osm_city_name).find_word_form(df, street_names)

        del street_names
        gdf = self.create_gdf(df)

        gdf = pd.concat([gdf, df_obj], ignore_index=True)
        del df_obj
        gdf["geo_obj_tag"] = gdf["geo_obj_tag"].apply(Geocoder.assign_street)

        gdf = pd.merge(gdf, initial_df, on=text_column, how='right')

        # gdf = pd.concat(
        #     [gdf, df.copy()[~df.copy()[text_column].isin(gdf[text_column])]], ignore_index=True
        # )

        # gdf2 = self.merge_to_initial_df(gdf, initial_df)

        # # Add a new 'level' column using the get_level function
        # gdf2["level"] = gdf2.progress_apply(self.get_level, axis=1)
        # gdf2 = self.set_global_repr_point(gdf2)
        gdf.set_crs(4326, inplace=True)
        return gdf
    

    # def extract_ner_street(self, text: str) -> pd.Series:
    #     """
    #     Function calls the pre-trained custom NER model to extract mentioned
    #     addresses from the texts (usually comment) in social networks in
    #     russian language.
    #     The model scores 0.8 in F1 and other metrics.
    #     """

    #     try:
    #         text = re.sub(r"\[.*?\]", "", text)
    #     except Exception:
    #         return pd.Series([None, None])

    #     sentence = Sentence(text)
    #     self.classifier.predict(sentence)
    #     try:
    #         res = sentence.get_labels("ner")[0].labeled_identifier.split("]: ")[1].split("/")[0].replace('"', "")
    #         score = round(sentence.get_labels("ner")[0].score, 3)
    #         if score > 0.7:
    #             return pd.Series([res, score])
    #         else:
    #             return pd.Series([None, None])

    #     except IndexError:
    #         return pd.Series([None, None])

    # @staticmethod
    # def extract_building_num(text, street_name, number) -> string:
    #     """
    #     The function finds the already extracted street name in the text
    #     and searches for numbers related to building numbers in a certain range of indexes
    #     around the street name.
    #     """
    #     if pd.notna(number) and number != "":
    #         return number
    #     if isinstance(text, float) and math.isnan(text):
    #         return ""

    #     clear_text = str(text).translate(str.maketrans("", "", string.punctuation))
    #     clear_text = clear_text.lower().split(" ")
    #     positions = [index for index, item in enumerate(clear_text) if item == street_name]

    #     if not positions:
    #         return ""

    #     position = positions[0]
    #     search_start = max(0, position)
    #     search_end = min(len(clear_text), position + END_INDEX_POSITION)

    #     num_result = []

    #     for f_index in range(max(0, search_start), min(len(clear_text), search_end)):
    #         element = clear_text[f_index]
    #         if any(character.isdigit() for character in str(element)) and len(str(element)) <= 3:
    #             num_result.append(element)
    #             break

    #     if num_result:
    #         return num_result[0]
    #     else:
    #         return ""

    # @staticmethod
    # def extract_toponym(text, street_name) -> string:
    #     """
    #     The function finds the already extracted street name in the text
    #     and searches for words related to toponyms in a certain range of indexes
    #     around the street name.
    #     """
    #     if isinstance(text, float) and math.isnan(text):
    #         return None

    #     clear_text = str(text).translate(str.maketrans("", "", string.punctuation))
    #     clear_text = clear_text.lower().split(" ")
    #     positions = [index for index, item in enumerate(clear_text) if item == street_name]

    #     if not positions:
    #         return None

    #     position = positions[0]
    #     search_start = max(0, position - int(START_INDEX_POSITION))
    #     search_end = min(len(clear_text), position + int(END_INDEX_POSITION))

    #     ad_result = []
    #     for i in range(search_start, min(search_end + 1, len(clear_text))):
    #         word = clear_text[i]
    #         normal_form = morph.parse(word)[0].normal_form
    #         if normal_form in TARGET_TOPONYMS:
    #             ad_result.append(REPLACEMENT_DICT.get(normal_form, normal_form))

    #     if ad_result:
    #         return ad_result[0]
    #     else:
    #         return None

       # def get_street(self, df: pd.DataFrame, text_column: str) -> gpd.GeoDataFrame:
    #     """
    #     Function calls NER model and post-process result in order to extract
    #     the address mentioned in the text.
    #     """
    #     logger.info("get_street started")

    #     df[text_column].dropna(inplace=True)
    #     df[text_column] = df[text_column].astype(str)

    #     logger.info("extract_ner_street started")

    #     df[["Street", "Score"]] = df[text_column].progress_apply(lambda t: self.extract_ner_street(t))
    #     df["Street"] = df[[text_column, "Street"]].progress_apply(
    #         lambda row: Geocoder.get_ner_address_natasha(row, text_column),
    #         axis=1,
    #     )

    #     df = df[df.Street.notna()]
    #     df = df[df["Street"].str.contains("[а-яА-Я]")]

    #     logger.info("pattern1.sub started")

    #     pattern1 = re.compile(r"(\D)(\d)(\D)")
    #     df["Street"] = df["Street"].progress_apply(lambda x: pattern1.sub(r"\1 \2\3", x))

    #     logger.info("pattern2.findall started")

    #     pattern2 = re.compile(r"\d+")
    #     df["Numbers"] = df["Street"].progress_apply(lambda x: " ".join(pattern2.findall(x)))

    #     logger.info("pattern2.sub started")

    #     df["Street"] = df["Street"].progress_apply(lambda x: pattern2.sub("", x).strip())

    #     df["initial_street"] = df["Street"].copy()

    #     df["Street"] = df["Street"].str.lower()

    #     logger.info("extract_building_num started")

    #     df["Numbers"] = df.progress_apply(
    #         lambda row: Geocoder.extract_building_num(row[text_column], row["Street"], row["Numbers"]),
    #         axis=1,
    #     )

    #     logger.info("extract_toponym started")

    #     df["Toponims"] = df.progress_apply(
    #         lambda row: Geocoder.extract_toponym(row[text_column], row["Street"]),
    #         axis=1,
    #     )
    #     return df

    # df = pd.DataFrame(data={'text': 'На биржевой 14 что-то произошло'}, index=[0])
    # print(Geocoder().run(df=df, text_column='text'))


    # def find_word_form(self, df: pd.DataFrame, strts_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     In Russian language any word has different forms.
    #     Since addresses are extracted from the texts in social networks,
    #     they might be in any possible form. This function is aimed to match that
    #     free form to the one that is used in the OSM database.

    #     Since the stem is found there would be several streets with that stem
    #     in their name.
    #     However, the searching street name has its specific ending (form) and
    #     not each matched street name could have it.
    #     """

    #     df["full_street_name"] = None

    #     for idx, row in df.iterrows():
    #         search_val = row["Street"]
    #         search_top = row["Toponims"]
    #         val_num = row["Numbers"]

    #         for col in strts_df.columns[2:]:
    #             search_rows = strts_df.loc[strts_df[col] == search_val]
    #             matching_rows = search_rows[search_rows["toponim_name"] == search_top]

    #             if not matching_rows.empty:
    #                 only_streets_full = matching_rows["street"].values
    #                 streets_full = [
    #                     street + f" {val_num}" + f" {self.osm_city_name}" + " Россия" for street in only_streets_full
    #                 ]

    #                 df.loc[idx, "full_street_name"] = ",".join(streets_full)
    #                 df.loc[idx, "only_full_street_name"] = ",".join(only_streets_full)

    #             else:
    #                 if search_val in strts_df[col].values:
    #                     only_streets_full = strts_df.loc[strts_df[col] == search_val, "street"].values
    #                     streets_full = [
    #                         street + f" {val_num}" + f" {self.osm_city_name}" + " Россия"
    #                         for street in only_streets_full
    #                     ]

    #                     df.loc[idx, "full_street_name"] = ",".join(streets_full)
    #                     df.loc[idx, "only_full_street_name"] = ",".join(only_streets_full)

    #     df.dropna(subset=["full_street_name", "only_full_street_name"], inplace=True)
    #     df["location_options"] = df["full_street_name"].str.split(",")
    #     df["only_full_street_name"] = df["only_full_street_name"].str.split(",")

    #     tmp_df_1 = df["location_options"].explode()
    #     tmp_df_1.name = "addr_to_geocode"
    #     tmp_df_2 = df["only_full_street_name"].explode()
    #     tmp_df_2.name = "only_full_street_name"
    #     new_df = tmp_df_1.to_frame().join(tmp_df_2.to_frame())

    #     df.drop(columns=["only_full_street_name"], inplace=True)
    #     df = df.merge(new_df, left_on=df.index, right_on=new_df.index)
    #     df.drop(columns=["key_0"], inplace=True)

    #     # new_df = df["only_full_street_name"].explode()
    #     # new_df.name = "only_full_street_name"
    #     # df.drop(columns=['key_0', 'only_full_street_name'], inplace=True)
    #     # df = df.merge(new_df, left_on=df.index, right_on=new_df.index)

    #     # print(df.head())
    #     df["only_full_street_name"] = df["only_full_street_name"].astype(str)
    #     df["location_options"] = df["location_options"].astype(str)

    #     return df

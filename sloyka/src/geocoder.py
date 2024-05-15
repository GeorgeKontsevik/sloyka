"""
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
from typing import List, Optional
import os
import flair
import geopandas as gpd
import networkx as nx
import osm2geojson
import osmnx as ox
import pandas as pd
import pymorphy2
import requests
import torch
import difflib
import string
import math
from sloyka.src.constants import (
    START_INDEX_POSITION,
    REPLACEMENT_DICT,
    TARGET_TOPONYMS,
    END_INDEX_POSITION,
)

from flair.data import Sentence
from flair.models import SequenceTagger
from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from tqdm import tqdm
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc,
)

from loguru import logger

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=-1)

segmenter = Segmenter()
morph_vocab = MorphVocab()
morph = pymorphy2.MorphAnalyzer()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
warnings.simplefilter(action="ignore", category=FutureWarning)

tqdm.pandas()


class Location:
    """
    This class is aimed to efficiently geocode addresses using Nominatim.
    Geocoded addresses are stored in the 'book' dictionary argument.
    Thus, if the address repeats -- it would be taken from the book.
    """

    max_tries = 3

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

        for _ in range(Location.max_tries):
            try:
                geocode = self.geolocator.geocode(
                    query, addressdetails=True, language="ru"
                )
                return geocode
            except GeocoderUnavailable:
                pass
        return None

    def query(self, address: str) -> Optional[List[float]]:
        """
        A function to query the address and return its geocode if available.
        
        :param address: A string representing the address to be queried.
        :return: An optional list of floats representing the geocode of the address, or None if not found.
        """
        if address not in self.book:
            query = f"{address}"
            res = self.geocode_with_retry(query)
            self.book[address] = res

        return self.book.get(address)


class Streets:
    """
    This class encapsulates functionality for retrieving street data
    for a specified city from OSM and processing it to extract useful
    information for geocoding purposes.
    """

    global_crs: int = 4326

    @staticmethod
    def get_city_bounds(
        osm_city_name: str, osm_city_level: int
    ) -> gpd.GeoDataFrame:
        """
        Method retrieves the boundary of a specified city from OSM
        using Overpass API and returns a GeoDataFrame representing
        the boundary as a polygon.
        """
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area[name="{osm_city_name}"]->.searchArea;
                (
                relation["admin_level"="{osm_city_level}"](area.searchArea);
                );
        out geom;
        """

        result = requests.get(
            overpass_url, params={"data": overpass_query}
        ).json()
        resp = osm2geojson.json2geojson(result)
        city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(
            Streets.global_crs
        )
        return city_bounds

    @staticmethod
    def get_drive_graph(city_bounds: gpd.GeoDataFrame) -> nx.MultiDiGraph:
        """
        Method uses the OSMnx library to retrieve the street network for a
        specified city and returns it as a NetworkX MultiDiGraph object, where
        each edge represents a street segment and each node represents
        an intersection.
        """

        G_drive = ox.graph_from_polygon(
            city_bounds.dissolve()["geometry"].squeeze(), network_type="drive"
        )

        return G_drive

    @staticmethod
    def graph_to_gdf(G_drive: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        Method converts the street network from a NetworkX MultiDiGraph object
        to a GeoDataFrame representing the edges (streets) with columns
        for street name, length, and geometry.
        """

        gdf = ox.graph_to_gdfs(G_drive, nodes=False)
        gdf["name"].dropna(inplace=True)
        gdf = gdf[["name", "length", "geometry"]]
        gdf.reset_index(inplace=True)
        gdf = gpd.GeoDataFrame(data=gdf, geometry="geometry")

        return gdf

    @staticmethod
    def get_street_names(gdf: gpd.GeoDataFrame):
        """
        Method extracts the unique street names from a
        GeoDataFrame of street segments.
        """

        names = set(gdf["name"].explode().dropna())
        df_streets = pd.DataFrame(names, columns=["street"])

        return df_streets

    @staticmethod
    def find_toponim_words_from_name(x: str) -> str:
        pattern = [
            "путепровод", "улица", "набережная реки", "проспект",
            "бульвар", "мост", "переулок", "площадь", "переулок",
            "набережная", "канала", "канал", "дорога на", "дорога в",
            "шоссе", "аллея", "проезд", "линия"
        ]

        best_match_ratio = 0
        best_match = None

        for word in pattern:
            ratio = difflib.SequenceMatcher(None, word, x).ratio()
            if ratio >= 0.4 and ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match = word

        return best_match.lower() if best_match else None

    @staticmethod
    def drop_words_from_name(x: str) -> str:
        """
        This function drops parts of street names that are not the name
        of the street (e.g. avenue).
        """

        try:
            lst = re.split(
                r"путепровод|улица|набережная реки|проспект"
                r"|бульвар|мост|переулок|площадь|переулок"
                r"|набережная|канала|канал|дорога на|дорога в"
                r"|шоссе|аллея|проезд",
                x,
            )
            lst.remove("")

            return lst[0].strip().lower()

        except ValueError:
            return x

    @staticmethod
    def clear_names(streets_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function pre-process the street names from the OSM.
        This step is necessary to match recognised street addresses later.
        We need to do this match because Nominatim is very sensitive geocoder
        and requires almost exact match between addresses in the OSM database
        and the geocoding address.
        """
        logger.info('clear_names started')
        streets_df["toponim_name"] = streets_df["street"].map(
            Streets.find_toponim_words_from_name
        )
        streets_df["street_name"] = streets_df["street"].map(
            Streets.drop_words_from_name
        )
        return streets_df

    @staticmethod
    def run(osm_city_name: str, osm_city_level: int) -> pd.DataFrame:
        """
        A static method to run the process of getting street data based on the given
        OSM city name and level, returning a pandas DataFrame.
        """
        city_bounds = Streets.get_city_bounds(osm_city_name, osm_city_level)
        streets_graph = Streets.get_drive_graph(city_bounds)
        streets_gdf = Streets.graph_to_gdf(streets_graph)
        streets_df = Streets.get_street_names(streets_gdf)
        streets_df = Streets.clear_names(streets_df)

        return streets_df


class Geocoder:
    """
    This class provides a functionality of simple geocoder
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    global_crs: int = 4326
    exceptions = pd.merge(
        pd.read_csv(
            os.path.join(dir_path, "exceptions_countries.csv"),
            encoding="utf-8",
            sep=",",
        ),
        pd.read_csv(
            os.path.join(dir_path, "exсeptions_city.csv"),
            encoding="utf-8",
            sep=",",
        ),
        on="Сокращенное наименование",
        how="outer",
    )
    global_crs: int = 4326

    def __init__(
        self,
        model_path: str = "Geor111y/flair-ner-addresses-extractor",
        device: str = "cpu",
        osm_city_level: int = 5,
        osm_city_name: str = "Санкт-Петербург",
    ):
        self.device = device
        flair.device = torch.device(device)
        self.classifier = SequenceTagger.load(model_path)
        self.osm_city_level = osm_city_level
        self.osm_city_name = osm_city_name

    def extract_ner_street(self, text: str) -> pd.Series:
        """
        Function calls the pre-trained custom NER model to extract mentioned
        addresses from the texts (usually comment) in social networks in
        russian language.
        The model scores 0.8 in F1 and other metrics.
        """

        try:
            text = re.sub(r"\[.*?\]", "", text)
        except Exception:
            return pd.Series([None, None])

        sentence = Sentence(text)
        self.classifier.predict(sentence)
        try:
            res = (
                sentence.get_labels("ner")[0]
                .labeled_identifier.split("]: ")[1]
                .split("/")[0]
                .replace('"', "")
            )
            score = round(sentence.get_labels("ner")[0].score, 3)
            if score > 0.7:
                return pd.Series([res, score])
            else:
                return pd.Series([None, None])

        except IndexError:
            return pd.Series([None, None])

    @staticmethod
    def get_ner_address_natasha(row, exceptions, text_col) -> string or None:
        """
        The function extracts street names in the text, using the Natasha library,
        in cases where BERT could not.
        """
        try:
            if row["Street"] == None or row["Street"] == np.nan:
                i = row[text_col]
                location_final = []
                i = re.sub(r"\[.*?\]", "", i)
                doc = Doc(i)
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)
                doc.parse_syntax(syntax_parser)
                doc.tag_ner(ner_tagger)
                for span in doc.spans:
                    span.normalize(morph_vocab)
                location = list(filter(lambda x: x.type == "LOC", doc.spans))
                for span in location:
                    if (
                        span.normal.lower()
                        not in exceptions["Сокращенное наименование"]
                        .str.lower()
                        .values
                    ):
                        location_final.append(span)
                location_final = [(span.text) for span in location_final]
                if not location_final:
                    return None
                return location_final[0]
            else:
                return row["Street"]
            
        except ValueError:
            return None

    @staticmethod
    def extract_building_num(text, street_name, number) -> string:
        """
        The function finds the already extracted street name in the text
        and searches for numbers related to building numbers in a certain range of indexes
        around the street name.
        """
        if pd.notna(number) and number != "":
            return number
        if isinstance(text, float) and math.isnan(text):
            return ""

        clear_text = str(text).translate(
            str.maketrans("", "", string.punctuation)
        )
        clear_text = clear_text.lower().split(" ")
        positions = [
            index
            for index, item in enumerate(clear_text)
            if item == street_name
        ]

        if not positions:
            return ""

        position = positions[0]
        search_start = max(0, position)
        search_end = min(len(clear_text), position + END_INDEX_POSITION)

        num_result = []

        for f_index in range(
            max(0, search_start), min(len(clear_text), search_end)
        ):
            element = clear_text[f_index]
            if (
                any(character.isdigit() for character in str(element))
                and len(str(element)) <= 3
            ):
                num_result.append(element)
                break

        if num_result:
            return num_result[0]
        else:
            return ""

    @staticmethod
    def extract_toponym(text, street_name) -> string:
        """
        The function finds the already extracted street name in the text
        and searches for words related to toponyms in a certain range of indexes
        around the street name.
        """
        if isinstance(text, float) and math.isnan(text):
            return None

        clear_text = str(text).translate(
            str.maketrans("", "", string.punctuation)
        )
        clear_text = clear_text.lower().split(" ")
        positions = [
            index
            for index, item in enumerate(clear_text)
            if item == street_name
        ]

        if not positions:
            return None

        position = positions[0]
        search_start = max(0, position - int(START_INDEX_POSITION))
        search_end = min(len(clear_text), position + int(END_INDEX_POSITION))

        ad_result = []
        for i in range(search_start, min(search_end + 1, len(clear_text))):
            word = clear_text[i]
            normal_form = morph.parse(word)[0].normal_form
            if normal_form in TARGET_TOPONYMS:
                ad_result.append(REPLACEMENT_DICT.get(normal_form, normal_form))

        if ad_result:
            return ad_result[0]
        else:
            return None

    @staticmethod
    def get_stem(street_names_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function finds the stem of the word to find this stem in the street
        names dictionary (df).
        """
        logger.info('get_stem started')

        morph = pymorphy2.MorphAnalyzer()
        cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

        for case in cases:
            street_names_df[case] = street_names_df[
                "street_name"
            ].apply(
                lambda x: morph.parse(x)[0].inflect({case}).word
                if morph.parse(x)[0].inflect({case})
                else None
            )
        return street_names_df
    
    @staticmethod
    def _join_only_full_street_name_numbers(x):
        street_name = x['only_full_street_name']
        house_num = x['Numbers']
        if street_name and house_num:
            return street_name + ' ' + house_num
        elif street_name and not house_num:
            return street_name
        elif not street_name and house_num:
            return None
        else:
            return None

    def find_word_form(
        self, df: pd.DataFrame, strts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        In the Russian language, any word has different forms.
        Since addresses are extracted from the texts in social networks,
        they might be in any possible form. This function is aimed to match that
        free form to the one that is used in the OSM database.

        Since the stem is found there would be several streets with that stem
        in their name.
        However, the searching street name has its specific ending (form) and
        not each matched street name could have it.

        TODO: add spellchecker since there might be misspelled words.
        """

        df["full_street_name"] = None

        for idx, row in df.iterrows():
            search_val = row["Street"]
            search_top = row["Toponims"]
            val_num = row["Numbers"]

            for col in strts_df.columns[2:]:
                search_rows = strts_df.loc[strts_df[col] == search_val]
                matching_rows = search_rows[
                    search_rows["toponim_name"] == search_top
                ]

                if not matching_rows.empty:
                    only_streets_full = matching_rows["street"].values
                    streets_full = [
                        street
                        + f" {val_num}"
                        + f" {self.osm_city_name}"
                        + " Россия"
                        for street in only_streets_full
                    ]
                
                    df.loc[idx, "full_street_name"] = ",".join(streets_full)
                    df.loc[idx, "only_full_street_name"] = ",".join(only_streets_full)

                else:
                    if search_val in strts_df[col].values:
                        only_streets_full = strts_df.loc[
                            strts_df[col] == search_val, "street"
                        ].values
                        streets_full = [
                            street
                            + f" {val_num}"
                            + f" {self.osm_city_name}"
                            + " Россия"
                            for street in only_streets_full
                        ]

                        df.loc[idx, "full_street_name"] = ",".join(streets_full)
                        df.loc[idx, "only_full_street_name"] = ",".join(only_streets_full)


        # df.dropna(subset=["full_street_name", 'only_full_street_name'], inplace=True)
        
        df["location_options"] = df["full_street_name"].str.split(",")
        df["only_full_street_name"] = df["only_full_street_name"].str.split(",")
        

        tmp_df_1 = df["location_options"].explode()
        tmp_df_1.name = "addr_to_geocode"
        tmp_df_2 = df["only_full_street_name"].explode()
        tmp_df_2.name = "only_full_street_name"
        new_df = tmp_df_1.to_frame().join(tmp_df_2.to_frame()) 

        df.drop(columns=['only_full_street_name'], inplace=True)
        df = df.merge(new_df, left_index=True, right_index=True)
        logger.debug(df)
        # df.drop(columns=['key_0'], inplace=True)

        # new_df = df["only_full_street_name"].explode()
        # new_df.name = "only_full_street_name"
        # df.drop(columns=['key_0', 'only_full_street_name'], inplace=True)
        # df = df.merge(new_df, left_on=df.index, right_on=new_df.index)

        # print(df.head())
        df["only_full_street_name"] = df["only_full_street_name"].astype(str)
        df["location_options"] = df["location_options"].astype(str)
        df['only_full_street_name'] = df['only_full_street_name'].map(lambda x: None if x == 'nan' or x == '' else x)
        df['only_full_street_name_numbers'] = df.apply(Geocoder._join_only_full_street_name_numbers, axis=1)
        return df

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

    def get_street(
        self, df: pd.DataFrame, text_column: str
    ) -> gpd.GeoDataFrame:
        """
        Function calls NER model and post-process result in order to extract
        the address mentioned in the text.
        """
        logger.info('get_street started')

        # df[text_column].dropna(inplace=True)
        df[text_column] = df[text_column].astype(str)
        
        logger.info('extract_ner_street started')

        df[["Street", "Score"]] = df[text_column].progress_apply(
            lambda t: self.extract_ner_street(t)
        )
        df["Street"] = df[[text_column, "Street"]].progress_apply(
            lambda row: Geocoder.get_ner_address_natasha(
                row, self.exceptions, text_column
            ) if ["Street"] else None,
            axis=1,
        )

        # df = df[df.Street.notna()]
        # df = df[df["Street"].str.contains("[а-яА-Я]")]

        logger.info('pattern1.sub started')

        pattern1 = re.compile(r"(\D)(\d)(\D)")
        df["Street"] = df["Street"].progress_apply(lambda x: pattern1.sub(r"\1 \2\3", x) if x else x)

        logger.info('pattern2.findall started')

        pattern2 = re.compile(r"\d+")
        df["Numbers"] = df["Street"].progress_apply(
            lambda x: " ".join(pattern2.findall(x)) if x else x
        )


        logger.info('pattern2.sub started')


        df["Street"] = df["Street"].progress_apply(lambda x: pattern2.sub("", x).strip() if x else x)

        df['initial_street'] = df['Street'].copy()

        df["Street"] = df["Street"].str.lower()

        logger.info('extract_building_num started')

        df["Numbers"] = df.progress_apply(
            lambda row: Geocoder.extract_building_num(
                row[text_column], row["Street"], row["Numbers"]
            ),
            axis=1,
        )

        logger.info('extract_toponym started')


        df["Toponims"] = df.progress_apply(
            lambda row: Geocoder.extract_toponym(
                row[text_column], row["Street"]
            ),
            axis=1,
        )
        return df

    def create_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Function simply creates gdf from the recognised geocoded geometries.
        """
        logger.info('create_gdf started')


        df["Location"] = df["addr_to_geocode"].progress_apply(lambda x: Location().query(x) if x else x)
        # df = df.dropna(subset=["Location"])
        df["geometry"] = df.Location.apply(
            lambda x: Point(x.longitude, x.latitude) if x else None
        )
        df["Location"] = df.Location.apply(lambda x: x.address if x else None)
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

    def merge_to_initial_df(
        self, gdf: gpd.GeoDataFrame, initial_df: pd.DataFrame
    ) -> gpd.GeoDataFrame:
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
        gdf = gpd.GeoDataFrame(
            gdf, geometry="geometry", crs=Geocoder.global_crs
        )

        return gdf

    def run(self, df: pd.DataFrame, text_column: str = "Текст комментария"):
        """
        Runs the data processing pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the text column in the DataFrame. Defaults to "Текст комментария".

        Returns:
            gpd.GeoDataFrame: The processed DataFrame after running the data processing pipeline.
        """
        # initial_df = df.copy()
        street_names = Streets.run(self.osm_city_name, self.osm_city_level)

        df = self.get_street(df, text_column)
        street_names = self.get_stem(street_names)
        df = self.find_word_form(df, street_names)
        gdf = self.create_gdf(df)
        # gdf2 = self.merge_to_initial_df(gdf, initial_df)

        # # Add a new 'level' column using the get_level function
        # gdf2["level"] = gdf2.progress_apply(self.get_level, axis=1)
        # gdf2 = self.set_global_repr_point(gdf2)

        return gdf

if __name__ == '__main__':
    df = pd.DataFrame(data={'text': 'На биржевой 14 что-то произошло'}, index=[0])
    print(Geocoder().run(df=df, text_column='text'))
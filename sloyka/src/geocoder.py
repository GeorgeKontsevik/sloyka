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
from shapely.geometry import Point, Polygon, MultiPolygon
import pdb
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
import string
import math
from rapidfuzz import fuzz
from nltk.stem.snowball import SnowballStemmer
from sloyka.src.data_getter import HistGeoDataGetter
from sloyka.src.constants import (
    START_INDEX_POSITION,
    REPLACEMENT_DICT,
    TARGET_TOPONYMS,
    END_INDEX_POSITION,
    NUM_CITY_OBJ,
    EXCEPTIONS_CITY_COUNTRY,
    AREA_STOPWORDS,
    GROUP_STOPWORDS,
    REGEX_PATTERN,
    REPLACEMENT_STRING
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
from .rule_for_natasha import ADDR_PART
from natasha.extractors import Match
from natasha.extractors import Extractor
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
warnings.filterwarnings("ignore", category=DeprecationWarning)

stemmer = SnowballStemmer("russian")

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
        """
        A method to find toponim words from the given name string.

        Args:
            x (str): The input name string.

        Returns:
            str: The found toponim word from the input name string, or None if not found.
        """
        pattern = re.compile(
            r"путепровод|улица|набережная реки|проспект"
            r"|бульвар|мост|переулок|площадь|переулок"
            r"|набережная|канала|канал|дорога на|дорога в"
            r"|шоссе|аллея|проезд|линия",
            re.IGNORECASE,
        )

        match = pattern.search(x)

        if match:
            return match.group().strip().lower()
        else:
            return None

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

class AddrNEWExtractor(Extractor):
    def __init__(self, morph):
        Extractor.__init__(self, ADDR_PART, morph)

    def find(self, text):
        matches = self(text)
        if not matches:
            return

        matches = sorted(matches, key=lambda _: _.start)
        if not matches:
            return
        start = matches[0].start
        stop = matches[-1].stop
        parts = [_.fact for _ in matches]
        return Match(start, stop, obj.Addr(parts)) # type: ignore
class Other_geo_objects:
    @staticmethod
    def get_OSM_green_obj(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about green_obj.
        """
        tags = {'leisure': ['park', 'garden', 'recreation_ground']}
        green_obj = ox.geometries_from_place(osm_city_name, tags)
        osm_green_obj_df = pd.DataFrame(green_obj)
        osm_green_obj_df = osm_green_obj_df.dropna(subset=['name'])
        osm_green_obj_df = osm_green_obj_df[['name', 'geometry', 'leisure']]
        return osm_green_obj_df
    
    @staticmethod
    def get_OSM_num_obj(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about amenity.
        """
        tags = {'amenity': ['hospital', 'clinic', 'school', 'kindergarten']}
        osm_num_obj = ox.geometries_from_place(osm_city_name, tags)
        osm_num_obj_df = pd.DataFrame(osm_num_obj)
        osm_num_obj_df = osm_num_obj_df.dropna(subset=['name'])
        osm_num_obj_df = osm_num_obj_df[['name', 'geometry', 'amenity']]
        return osm_num_obj_df
    
    @staticmethod
    def get_OSM_cemetery(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about cemetery.
        """
        tags = {'landuse': ['cemetery']}
        osm_cemetery = ox.geometries_from_place(osm_city_name, tags)
        osm_cemetery_df = pd.DataFrame(osm_cemetery)
        osm_cemetery_df = osm_cemetery_df.dropna(subset=['name'])
        osm_cemetery_df = osm_cemetery_df[['name', 'geometry', 'landuse']]
        return osm_cemetery_df
    
    @staticmethod
    def get_OSM_natural(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about natural obj.
        """
        tags = {'natural': ['beach', 'water']}
        osm_natural = ox.geometries_from_place(osm_city_name, tags)
        osm_natural_df = pd.DataFrame(osm_natural)
        osm_natural_df = osm_natural_df.dropna(subset=['name'])
        osm_natural_df = osm_natural_df[['name', 'geometry', 'natural']]
        return osm_natural_df
    
    @staticmethod
    def get_OSM_railway(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about railway obj.
        """
        tags = {'railway': ['station', 'subway']}
        osm_railway = ox.geometries_from_place(osm_city_name, tags)
        osm_railway_df = pd.DataFrame(osm_railway)
        osm_railway_df = osm_railway_df.dropna(subset=['name'])
        osm_railway_df = osm_railway_df[['name', 'geometry', 'railway']]
        return osm_railway_df
    
    @staticmethod
    def get_OSM_tourism(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about tourism obj.
        """
        tags = {'tourism': ['attraction', 'museum']}
        osm_tourism = ox.geometries_from_place(osm_city_name, tags)
        osm_tourism_df = pd.DataFrame(osm_tourism)
        osm_tourism_df = osm_tourism_df.dropna(subset=['name'])
        osm_tourism_df = osm_tourism_df[['name', 'geometry', 'tourism']]
        return osm_tourism_df
    
    @staticmethod
    def get_OSM_historic(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about historical obj.
        """
        tags = {'historic': ['monument', 'memorial']}
        osm_historic = ox.geometries_from_place(osm_city_name, tags)
        osm_historic_df = pd.DataFrame(osm_historic)
        osm_historic_df = osm_historic_df.dropna(subset=['name'])
        osm_historic_df = osm_historic_df[['name', 'geometry', 'historic']]
        return osm_historic_df

    @staticmethod    
    def get_OSM_square(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about square obj.
        """
        tags = {'place': ['square']}
        osm_square = ox.geometries_from_place(osm_city_name, tags)
        osm_square_df = pd.DataFrame(osm_square)
        osm_square_df = osm_square_df.dropna(subset=['name'])
        osm_square_df = osm_square_df[['name', 'geometry', 'place']]
        return osm_square_df

    @staticmethod
    def calculate_centroid(geometry) -> pd.DataFrame:
        """
        This function counts the centroid for polygons.
        """
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry.centroid
        elif isinstance(geometry, Point):
            return geometry
        else:
            return None
    
    def get_and_process_osm_data(osm_city_name, get_data_function) -> pd.DataFrame:
        """
        This function allows you to build an OSM array for different urban objects.
        """
        df = get_data_function(osm_city_name)
        df['geometry'] = df['geometry'].apply(Other_geo_objects.calculate_centroid)
        df.rename(columns={df.columns[2]: 'geo_obj_tag'}, inplace=True)
        return df
        
    def run_OSM_dfs(osm_city_name) -> pd.DataFrame:
        """
        This function collects dataframes with OSM spatial data, finds centroids and combines files into one.
        """
        logger.info('run_OSM_dfs started')

        osm_functions = [
            Other_geo_objects.get_OSM_green_obj,
            Other_geo_objects.get_OSM_num_obj,
            Other_geo_objects.get_OSM_cemetery,
            Other_geo_objects.get_OSM_natural,
            Other_geo_objects.get_OSM_railway,
            Other_geo_objects.get_OSM_tourism,
            Other_geo_objects.get_OSM_historic
        ]

        osm_dfs = [Other_geo_objects.get_and_process_osm_data(osm_city_name, func) for func in osm_functions]
        osm_combined_df = pd.concat(osm_dfs, axis=0)

        return osm_combined_df
        
    @staticmethod
    def extract_geo_obj(text) -> List[str]:
        """
        The function extracts location entities from the text, using the Natasha library.
        """
        morph = MorphVocab()
        extractor = AddrNEWExtractor(morph)

        other_geo_obj = []

        matches = extractor(text)
        for match in matches:
            part = match.fact
            if part.value and part.type:
                combined_phrase = f"{part.value} {part.type}"
                other_geo_obj.append(combined_phrase)
            elif part.value:
                other_geo_obj.append(part.value)
            elif part.type:
                other_geo_obj.append(part.type)

        return other_geo_obj

    def restoration_of_normal_form(other_geo_obj, osm_combined_df, threshold=0.7) -> List[str]:
        """
        This function compares the extracted location entity with an OSM array and returns a normalized form if the percentage of similarity is at least 70%.
        """
        osm_name_obj = osm_combined_df['name'].tolist()
        similarity_matrix = np.zeros((len(other_geo_obj), len(osm_name_obj)))
        
        def extract_numbers(s):
            return re.findall(r'\d+', s)

        for i, word1 in enumerate(other_geo_obj):
            numbers_from_extraction = extract_numbers(word1)
            for j, word2 in enumerate(osm_name_obj):
                numbers_from_OSM_name = extract_numbers(word2)
                if numbers_from_extraction == numbers_from_OSM_name:
                    similarity = fuzz.ratio(word1, word2) / 100.0
                else:
                    similarity = 0  
                similarity_matrix[i, j] = similarity
        
        restoration_list = other_geo_obj.copy()
        for i in range(len(other_geo_obj)):
            max_index = np.argmax(similarity_matrix[i])
            if similarity_matrix[i, max_index] > threshold:
                restoration_list[i] = osm_name_obj[max_index]
            else:
                restoration_list[i] = ""
        
        return restoration_list
    
    @staticmethod
    def find_num_city_obj(text, NUM_CITY_OBJ) -> List[str]:
        """
        This function searches for urban objects in the text, the names of which are represented as a number. For example, "school No. 6".
        """
        text = text.lower()
        num_obj_list = []
        for key, forms in NUM_CITY_OBJ.items():
            for form in forms:
                pattern = fr'\b{re.escape(form)}\b\s+№?\s*(\d+)'
                matches = re.findall(pattern, text)
                for match in matches:
                    num_obj_list.append(f"{key} № {match}")
        num_obj_list = list(set(num_obj_list))
        num_obj_list_clear = {}
        for obj in num_obj_list:
            key = obj.split(' № ')[1]
            if key in num_obj_list_clear:
                if len(obj.split(' № ')[0]) > len(num_obj_list_clear[key].split(' № ')[0]):
                    num_obj_list_clear[key] = obj
            else:
                num_obj_list_clear[key] = obj
        
        return list(num_obj_list_clear.values())
    
    @staticmethod
    def combine_city_obj(df_obj) -> pd.DataFrame:
        """
        This function combines the found named urban objects and urban objects whose names are in the form of numbers.
        """
        df_obj['other_geo_obj'] = df_obj['other_geo_obj'] + df_obj['other_geo_obj_num']
        df_obj.drop(columns=['other_geo_obj_num'], inplace=True)
        return df_obj
    
    @staticmethod
    def expand_toponim(df_obj) -> pd.DataFrame:
        """
        This function splits the list of found entities into different rows for further analysis.
        """
        expanded_df = df_obj.copy()
        expanded_df['other_geo_obj'] = expanded_df['other_geo_obj'].apply(lambda x: x if isinstance(x, list) and x else None)
        expanded_df = expanded_df.explode('other_geo_obj').reset_index(drop=True)
        return expanded_df
    
    @staticmethod
    def find_geometry(toponim, osm_combined_df) -> List[str]:
        """
        This function finds the coordinate in the OSM array by the name of the city object.
        """
        if toponim is None:
            return None
        match = osm_combined_df[osm_combined_df['name'] == toponim]
        if not match.empty:
            geometry = match.iloc[0, 1]
            return geometry
        else:
            return None
    
    @staticmethod
    def find_geo_obj_tag(toponim, osm_combined_df) -> List[str]:
        """
        This function finds the geo_obj_tag in the OSM array by the name of the city object.
        """
        if toponim is None:
            return None
        match = osm_combined_df[osm_combined_df['name'] == toponim]
        if not match.empty:
            leisure = match.iloc[0, 2]
            return leisure
        else:
            return None

    
    @staticmethod
    def run(osm_city_name, df, text_column) -> pd.DataFrame:
        """
        This function launches the module for extracting urban objects from texts that do not relate to streets.
        """
        df_obj = df.copy()
        osm_combined_df = Other_geo_objects.run_OSM_dfs(osm_city_name)
        logger.info('find_other_geo_obj started')
        df_obj['other_geo_obj'] = df_obj[text_column].apply(Other_geo_objects.extract_geo_obj)
        df_obj['other_geo_obj_num'] = df_obj[text_column].apply(lambda x: Other_geo_objects.find_num_city_obj(x, NUM_CITY_OBJ))
        df_obj = Other_geo_objects.combine_city_obj(df_obj)
        df_obj['other_geo_obj'] = df_obj['other_geo_obj'].apply(lambda x: Other_geo_objects.restoration_of_normal_form(x, osm_combined_df))
        df_obj = Other_geo_objects.expand_toponim(df_obj)
        df_obj['geometry'] = df_obj['other_geo_obj'].apply(lambda x: Other_geo_objects.find_geometry(x, osm_combined_df))
        df_obj['geo_obj_tag'] = df_obj['other_geo_obj'].apply(lambda x: Other_geo_objects.find_geo_obj_tag(x, osm_combined_df))
        df_obj = df_obj[df_obj['geometry'].notna()]
        return df_obj

class Geocoder:
    """
    This class provides a functionality of simple geocoder
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

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
    def get_ner_address_natasha(row, EXCEPTIONS_CITY_COUNTRY, text_col) -> string:
        """
        The function extracts street names in the text, using the Natasha library,
        in cases where BERT could not.
        """
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
                if span.normal.lower() not in [name.lower() for name in EXCEPTIONS_CITY_COUNTRY]:
                    location_final.append(span)
            location_final = [(span.text) for span in location_final]
            if not location_final:
                return None
            return location_final[0]
        else:
            return row["Street"]

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


        df.dropna(subset=["full_street_name", 'only_full_street_name'], inplace=True)
        df["location_options"] = df["full_street_name"].str.split(",")
        df["only_full_street_name"] = df["only_full_street_name"].str.split(",")

        tmp_df_1 = df["location_options"].explode()
        tmp_df_1.name = "addr_to_geocode"
        tmp_df_2 = df["only_full_street_name"].explode()
        tmp_df_2.name = "only_full_street_name"
        new_df = tmp_df_1.to_frame().join(tmp_df_2.to_frame()) 

        df.drop(columns=['only_full_street_name'], inplace=True)
        df = df.merge(new_df, left_on=df.index, right_on=new_df.index)
        df.drop(columns=['key_0'], inplace=True)

        # new_df = df["only_full_street_name"].explode()
        # new_df.name = "only_full_street_name"
        # df.drop(columns=['key_0', 'only_full_street_name'], inplace=True)
        # df = df.merge(new_df, left_on=df.index, right_on=new_df.index)

        # print(df.head())
        df["only_full_street_name"] = df["only_full_street_name"].astype(str)
        df["location_options"] = df["location_options"].astype(str)

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

        df[text_column].dropna(inplace=True)
        df[text_column] = df[text_column].astype(str)
        
        logger.info('extract_ner_street started')

        df[["Street", "Score"]] = df[text_column].progress_apply(
            lambda t: self.extract_ner_street(t)
        )
        df["Street"] = df[[text_column, "Street"]].progress_apply(
            lambda row: Geocoder.get_ner_address_natasha(
                row, EXCEPTIONS_CITY_COUNTRY, text_column
            ),
            axis=1,
        )

        df = df[df.Street.notna()]
        df = df[df["Street"].str.contains("[а-яА-Я]")]

        logger.info('pattern1.sub started')

        pattern1 = re.compile(r"(\D)(\d)(\D)")
        df["Street"] = df["Street"].progress_apply(lambda x: pattern1.sub(r"\1 \2\3", x))

        logger.info('pattern2.findall started')

        pattern2 = re.compile(r"\d+")
        df["Numbers"] = df["Street"].progress_apply(
            lambda x: " ".join(pattern2.findall(x))
        )


        logger.info('pattern2.sub started')


        df["Street"] = df["Street"].progress_apply(lambda x: pattern2.sub("", x).strip())

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


        df["Location"] = df["addr_to_geocode"].progress_apply(Location().query)
        df = df.dropna(subset=["Location"])
        df["geometry"] = df.Location.apply(
            lambda x: Point(x.longitude, x.latitude)
        )
        df["Location"] = df.Location.apply(lambda x: x.address)
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

        return 
    
    def assign_street(variable):
        if isinstance(variable, float) and math.isnan(variable):
            return "street"
        return variable
    
    def get_df_areas(self, osm_id, tags, date):
        
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
            df_areas = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags, date=date)
            df_areas = df_areas[df_areas['element_type'] != 'way']
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
            group_name = re.sub(word, '', group_name, flags=re.IGNORECASE)
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
            df_areas['area_name'] = df_areas['name'].str.replace(word, '', regex=True)

        df_areas['area_name_processed'] = df_areas['area_name'].str.lower()
        df_areas['area_name_processed'] = df_areas['area_name_processed'].str.replace(REGEX_PATTERN, REPLACEMENT_STRING, regex=True)
        df_areas['area_stems'] = df_areas['area_name_processed'].apply(lambda x: [stemmer.stem(word) for word in x.split()])
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
            area_stems = row['area_stems']

            partial_ratio = fuzz.partial_ratio(group_name, row['area_name_processed'])
            token_sort_ratio = fuzz.token_sort_ratio(group_name_stems, area_stems)

            if partial_ratio > max_partial_ratio and token_sort_ratio > max_token_sort_ratio:
                max_partial_ratio = partial_ratio
                max_token_sort_ratio = token_sort_ratio
                best_match = row['area_name']
                admin_level = row['key']

        return best_match, admin_level
        

    def run(self, osm_id, tags, date, df: pd.DataFrame, text_column: str = "text", group_column: str = "group_name"):
        """
        Runs the data processing pipeline on the input DataFrame.

        Args:
            osm_id (int): The OpenStreetMap ID.
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
        
            # initial_df = df.copy()
        
        
        df_areas = self.get_df_areas(osm_id, tags, date)
        df_areas = self.preprocess_area_names(df_areas)

        for i, group_name in enumerate(df[group_column]):
            processed_group_name = self.preprocess_group_name(group_name)
            best_match, admin_level = self.match_group_to_area(processed_group_name, df_areas)
            df.at[i, 'territory'] = best_match
            df.at[i, 'admin_level'] = admin_level
        #df = AreaMatcher.run(self, df, osm_id, tags, date)

        df_obj = Other_geo_objects.run(self.osm_city_name, df, text_column)
        street_names = Streets.run(self.osm_city_name, self.osm_city_level)

        df = self.get_street(df, text_column)
        street_names = self.get_stem(street_names)
        df = self.find_word_form(df, street_names)
        gdf = self.create_gdf(df)
        gdf = pd.merge(gdf, df_obj, how='outer')
        gdf['geo_obj_tag'] = gdf['geo_obj_tag'].apply(Geocoder.assign_street)

            # gdf2 = self.merge_to_initial_df(gdf, initial_df)

            # # Add a new 'level' column using the get_level function
            # gdf2["level"] = gdf2.progress_apply(self.get_level, axis=1)
            # gdf2 = self.set_global_repr_point(gdf2)

        return gdf

if __name__ == '__main__':
    df = pd.DataFrame(data={'text': 'На биржевой 14 что-то произошло'}, index=[0])
    print(Geocoder().run(df=df, text_column='text'))
=======
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
from shapely.geometry import Point, Polygon, MultiPolygon
import pdb
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
import string
import math
from rapidfuzz import fuzz
from sloyka.src.constants import (
    START_INDEX_POSITION,
    REPLACEMENT_DICT,
    TARGET_TOPONYMS,
    END_INDEX_POSITION,
    NUM_CITY_OBJ,
    EXCEPTIONS_CITY_COUNTRY
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
from .rule_for_natasha import ADDR_PART
from natasha.extractors import Match
from natasha.extractors import Extractor
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
        """
        A method to find toponim words from the given name string.

        Args:
            x (str): The input name string.

        Returns:
            str: The found toponim word from the input name string, or None if not found.
        """
        pattern = re.compile(
            r"путепровод|улица|набережная реки|проспект"
            r"|бульвар|мост|переулок|площадь|переулок"
            r"|набережная|канала|канал|дорога на|дорога в"
            r"|шоссе|аллея|проезд|линия",
            re.IGNORECASE,
        )

        match = pattern.search(x)

        if match:
            return match.group().strip().lower()
        else:
            return None

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

class AddrNEWExtractor(Extractor):
    def __init__(self, morph):
        Extractor.__init__(self, ADDR_PART, morph)

    def find(self, text):
        matches = self(text)
        if not matches:
            return

        matches = sorted(matches, key=lambda _: _.start)
        if not matches:
            return
        start = matches[0].start
        stop = matches[-1].stop
        parts = [_.fact for _ in matches]
        return Match(start, stop, obj.Addr(parts))
class Other_geo_objects:
    @staticmethod
    def get_OSM_green_obj(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about green_obj.
        """
        tags = {'leisure': ['park', 'garden', 'recreation_ground']}
        green_obj = ox.geometries_from_place(osm_city_name, tags)
        osm_green_obj_df = pd.DataFrame(green_obj)
        osm_green_obj_df = osm_green_obj_df.dropna(subset=['name'])
        osm_green_obj_df = osm_green_obj_df[['name', 'geometry', 'leisure']]
        return osm_green_obj_df
    
    @staticmethod
    def get_OSM_num_obj(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about amenity.
        """
        tags = {'amenity': ['hospital', 'clinic', 'school', 'kindergarten']}
        osm_num_obj = ox.geometries_from_place(osm_city_name, tags)
        osm_num_obj_df = pd.DataFrame(osm_num_obj)
        osm_num_obj_df = osm_num_obj_df.dropna(subset=['name'])
        osm_num_obj_df = osm_num_obj_df[['name', 'geometry', 'amenity']]
        return osm_num_obj_df
    
    @staticmethod
    def get_OSM_cemetery(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about cemetery.
        """
        tags = {'landuse': ['cemetery']}
        osm_cemetery = ox.geometries_from_place(osm_city_name, tags)
        osm_cemetery_df = pd.DataFrame(osm_cemetery)
        osm_cemetery_df = osm_cemetery_df.dropna(subset=['name'])
        osm_cemetery_df = osm_cemetery_df[['name', 'geometry', 'landuse']]
        return osm_cemetery_df
    
    @staticmethod
    def get_OSM_natural(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about natural obj.
        """
        tags = {'natural': ['beach', 'water']}
        osm_natural = ox.geometries_from_place(osm_city_name, tags)
        osm_natural_df = pd.DataFrame(osm_natural)
        osm_natural_df = osm_natural_df.dropna(subset=['name'])
        osm_natural_df = osm_natural_df[['name', 'geometry', 'natural']]
        return osm_natural_df
    
    @staticmethod
    def get_OSM_railway(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about railway obj.
        """
        tags = {'railway': ['station', 'subway']}
        osm_railway = ox.geometries_from_place(osm_city_name, tags)
        osm_railway_df = pd.DataFrame(osm_railway)
        osm_railway_df = osm_railway_df.dropna(subset=['name'])
        osm_railway_df = osm_railway_df[['name', 'geometry', 'railway']]
        return osm_railway_df
    
    @staticmethod
    def get_OSM_tourism(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about tourism obj.
        """
        tags = {'tourism': ['attraction', 'museum']}
        osm_tourism = ox.geometries_from_place(osm_city_name, tags)
        osm_tourism_df = pd.DataFrame(osm_tourism)
        osm_tourism_df = osm_tourism_df.dropna(subset=['name'])
        osm_tourism_df = osm_tourism_df[['name', 'geometry', 'tourism']]
        return osm_tourism_df
    
    @staticmethod
    def get_OSM_historic(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about historical obj.
        """
        tags = {'historic': ['monument', 'memorial']}
        osm_historic = ox.geometries_from_place(osm_city_name, tags)
        osm_historic_df = pd.DataFrame(osm_historic)
        osm_historic_df = osm_historic_df.dropna(subset=['name'])
        osm_historic_df = osm_historic_df[['name', 'geometry', 'historic']]
        return osm_historic_df

    @staticmethod    
    def get_OSM_square(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about square obj.
        """
        tags = {'place': ['square']}
        osm_square = ox.geometries_from_place(osm_city_name, tags)
        osm_square_df = pd.DataFrame(osm_square)
        osm_square_df = osm_square_df.dropna(subset=['name'])
        osm_square_df = osm_square_df[['name', 'geometry', 'place']]
        return osm_square_df

    @staticmethod
    def calculate_centroid(geometry) -> pd.DataFrame:
        """
        This function counts the centroid for polygons.
        """
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry.centroid
        elif isinstance(geometry, Point):
            return geometry
        else:
            return None
    
    def get_and_process_osm_data(osm_city_name, get_data_function) -> pd.DataFrame:
        """
        This function allows you to build an OSM array for different urban objects.
        """
        df = get_data_function(osm_city_name)
        df['geometry'] = df['geometry'].apply(Other_geo_objects.calculate_centroid)
        df.rename(columns={df.columns[2]: 'geo_obj_tag'}, inplace=True)
        return df
        
    def run_OSM_dfs(osm_city_name) -> pd.DataFrame:
        """
        This function collects dataframes with OSM spatial data, finds centroids and combines files into one.
        """
        logger.info('run_OSM_dfs started')

        osm_functions = [
            Other_geo_objects.get_OSM_green_obj,
            Other_geo_objects.get_OSM_num_obj,
            Other_geo_objects.get_OSM_cemetery,
            Other_geo_objects.get_OSM_natural,
            Other_geo_objects.get_OSM_railway,
            Other_geo_objects.get_OSM_tourism,
            Other_geo_objects.get_OSM_historic
        ]

        osm_dfs = [Other_geo_objects.get_and_process_osm_data(osm_city_name, func) for func in osm_functions]
        osm_combined_df = pd.concat(osm_dfs, axis=0)

        return osm_combined_df
        
    @staticmethod
    def extract_geo_obj(text) -> List[str]:
        """
        The function extracts location entities from the text, using the Natasha library.
        """
        morph = MorphVocab()
        extractor = AddrNEWExtractor(morph)

        other_geo_obj = []

        matches = extractor(text)
        for match in matches:
            part = match.fact
            if part.value and part.type:
                combined_phrase = f"{part.value} {part.type}"
                other_geo_obj.append(combined_phrase)
            elif part.value:
                other_geo_obj.append(part.value)
            elif part.type:
                other_geo_obj.append(part.type)

        return other_geo_obj

    def restoration_of_normal_form(other_geo_obj, osm_combined_df, threshold=0.7) -> List[str]:
        """
        This function compares the extracted location entity with an OSM array and returns a normalized form if the percentage of similarity is at least 70%.
        """
        osm_name_obj = osm_combined_df['name'].tolist()
        similarity_matrix = np.zeros((len(other_geo_obj), len(osm_name_obj)))
        
        def extract_numbers(s):
            return re.findall(r'\d+', s)

        for i, word1 in enumerate(other_geo_obj):
            numbers_from_extraction = extract_numbers(word1)
            for j, word2 in enumerate(osm_name_obj):
                numbers_from_OSM_name = extract_numbers(word2)
                if numbers_from_extraction == numbers_from_OSM_name:
                    similarity = fuzz.ratio(word1, word2) / 100.0
                else:
                    similarity = 0  
                similarity_matrix[i, j] = similarity
        
        restoration_list = other_geo_obj.copy()
        for i in range(len(other_geo_obj)):
            max_index = np.argmax(similarity_matrix[i])
            if similarity_matrix[i, max_index] > threshold:
                restoration_list[i] = osm_name_obj[max_index]
            else:
                restoration_list[i] = ""
        
        return restoration_list
    
    @staticmethod
    def find_num_city_obj(text, NUM_CITY_OBJ) -> List[str]:
        """
        This function searches for urban objects in the text, the names of which are represented as a number. For example, "school No. 6".
        """
        text = text.lower()
        num_obj_list = []
        for key, forms in NUM_CITY_OBJ.items():
            for form in forms:
                pattern = fr'\b{re.escape(form)}\b\s+№?\s*(\d+)'
                matches = re.findall(pattern, text)
                for match in matches:
                    num_obj_list.append(f"{key} № {match}")
        num_obj_list = list(set(num_obj_list))
        num_obj_list_clear = {}
        for obj in num_obj_list:
            key = obj.split(' № ')[1]
            if key in num_obj_list_clear:
                if len(obj.split(' № ')[0]) > len(num_obj_list_clear[key].split(' № ')[0]):
                    num_obj_list_clear[key] = obj
            else:
                num_obj_list_clear[key] = obj
        
        return list(num_obj_list_clear.values())
    
    @staticmethod
    def combine_city_obj(df_obj) -> pd.DataFrame:
        """
        This function combines the found named urban objects and urban objects whose names are in the form of numbers.
        """
        df_obj['other_geo_obj'] = df_obj['other_geo_obj'] + df_obj['other_geo_obj_num']
        df_obj.drop(columns=['other_geo_obj_num'], inplace=True)
        return df_obj
    
    @staticmethod
    def expand_toponim(df_obj) -> pd.DataFrame:
        """
        This function splits the list of found entities into different rows for further analysis.
        """
        expanded_df = df_obj.copy()
        expanded_df['other_geo_obj'] = expanded_df['other_geo_obj'].apply(lambda x: x if isinstance(x, list) and x else None)
        expanded_df = expanded_df.explode('other_geo_obj').reset_index(drop=True)
        return expanded_df
    
    @staticmethod
    def find_geometry(toponim, osm_combined_df) -> List[str]:
        """
        This function finds the coordinate in the OSM array by the name of the city object.
        """
        if toponim is None:
            return None
        match = osm_combined_df[osm_combined_df['name'] == toponim]
        if not match.empty:
            geometry = match.iloc[0, 1]
            return geometry
        else:
            return None
    
    @staticmethod
    def find_geo_obj_tag(toponim, osm_combined_df) -> List[str]:
        """
        This function finds the geo_obj_tag in the OSM array by the name of the city object.
        """
        if toponim is None:
            return None
        match = osm_combined_df[osm_combined_df['name'] == toponim]
        if not match.empty:
            leisure = match.iloc[0, 2]
            return leisure
        else:
            return None

    
    @staticmethod
    def run(osm_city_name, df, text_column) -> pd.DataFrame:
        """
        This function launches the module for extracting urban objects from texts that do not relate to streets.
        """
        df_obj = df.copy()
        osm_combined_df = Other_geo_objects.run_OSM_dfs(osm_city_name)
        logger.info('find_other_geo_obj started')
        df_obj['other_geo_obj'] = df_obj[text_column].apply(Other_geo_objects.extract_geo_obj)
        df_obj['other_geo_obj_num'] = df_obj[text_column].apply(lambda x: Other_geo_objects.find_num_city_obj(x, NUM_CITY_OBJ))
        df_obj = Other_geo_objects.combine_city_obj(df_obj)
        df_obj['other_geo_obj'] = df_obj['other_geo_obj'].apply(lambda x: Other_geo_objects.restoration_of_normal_form(x, osm_combined_df))
        df_obj = Other_geo_objects.expand_toponim(df_obj)
        df_obj['geometry'] = df_obj['other_geo_obj'].apply(lambda x: Other_geo_objects.find_geometry(x, osm_combined_df))
        df_obj['geo_obj_tag'] = df_obj['other_geo_obj'].apply(lambda x: Other_geo_objects.find_geo_obj_tag(x, osm_combined_df))
        df_obj = df_obj[df_obj['geometry'].notna()]
        return df_obj


class Geocoder:
    """
    This class provides a functionality of simple geocoder
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

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
    def get_ner_address_natasha(row, EXCEPTIONS_CITY_COUNTRY, text_col) -> string:
        """
        The function extracts street names in the text, using the Natasha library,
        in cases where BERT could not.
        """
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
                if span.normal.lower() not in [name.lower() for name in EXCEPTIONS_CITY_COUNTRY]:
                    location_final.append(span)
            location_final = [(span.text) for span in location_final]
            if not location_final:
                return None
            return location_final[0]
        else:
            return row["Street"]

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


        df.dropna(subset=["full_street_name", 'only_full_street_name'], inplace=True)
        df["location_options"] = df["full_street_name"].str.split(",")
        df["only_full_street_name"] = df["only_full_street_name"].str.split(",")

        tmp_df_1 = df["location_options"].explode()
        tmp_df_1.name = "addr_to_geocode"
        tmp_df_2 = df["only_full_street_name"].explode()
        tmp_df_2.name = "only_full_street_name"
        new_df = tmp_df_1.to_frame().join(tmp_df_2.to_frame()) 

        df.drop(columns=['only_full_street_name'], inplace=True)
        df = df.merge(new_df, left_on=df.index, right_on=new_df.index)
        df.drop(columns=['key_0'], inplace=True)

        # new_df = df["only_full_street_name"].explode()
        # new_df.name = "only_full_street_name"
        # df.drop(columns=['key_0', 'only_full_street_name'], inplace=True)
        # df = df.merge(new_df, left_on=df.index, right_on=new_df.index)

        # print(df.head())
        df["only_full_street_name"] = df["only_full_street_name"].astype(str)
        df["location_options"] = df["location_options"].astype(str)

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

        df[text_column].dropna(inplace=True)
        df[text_column] = df[text_column].astype(str)
        
        logger.info('extract_ner_street started')

        df[["Street", "Score"]] = df[text_column].progress_apply(
            lambda t: self.extract_ner_street(t)
        )
        df["Street"] = df[[text_column, "Street"]].progress_apply(
            lambda row: Geocoder.get_ner_address_natasha(
                row, EXCEPTIONS_CITY_COUNTRY, text_column
            ),
            axis=1,
        )

        df = df[df.Street.notna()]
        df = df[df["Street"].str.contains("[а-яА-Я]")]

        logger.info('pattern1.sub started')

        pattern1 = re.compile(r"(\D)(\d)(\D)")
        df["Street"] = df["Street"].progress_apply(lambda x: pattern1.sub(r"\1 \2\3", x))

        logger.info('pattern2.findall started')

        pattern2 = re.compile(r"\d+")
        df["Numbers"] = df["Street"].progress_apply(
            lambda x: " ".join(pattern2.findall(x))
        )


        logger.info('pattern2.sub started')


        df["Street"] = df["Street"].progress_apply(lambda x: pattern2.sub("", x).strip())

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


        df["Location"] = df["addr_to_geocode"].progress_apply(Location().query)
        df = df.dropna(subset=["Location"])
        df["geometry"] = df.Location.apply(
            lambda x: Point(x.longitude, x.latitude)
        )
        df["Location"] = df.Location.apply(lambda x: x.address)
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

        return 
    
    def assign_street(variable):
        if isinstance(variable, float) and math.isnan(variable):
            return "street"
        return variable
    

    def run(self, df: pd.DataFrame, text_column: str = "Текст комментария"):
        """
        Runs the data processing pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the text column in the DataFrame. Defaults to "Текст комментария".

        Returns:
            pd.DataFrame: The processed DataFrame after running the data processing pipeline.
        """
        # initial_df = df.copy()
        df_obj = Other_geo_objects.run(self.osm_city_name, df, text_column)
        street_names = Streets.run(self.osm_city_name, self.osm_city_level)

        df = self.get_street(df, text_column)
        street_names = self.get_stem(street_names)
        df = self.find_word_form(df, street_names)
        gdf = self.create_gdf(df)
        gdf = pd.merge(gdf, df_obj, how='outer')
        gdf['geo_obj_tag'] = gdf['geo_obj_tag'].apply(Geocoder.assign_street)
        # gdf2 = self.merge_to_initial_df(gdf, initial_df)

        # # Add a new 'level' column using the get_level function
        # gdf2["level"] = gdf2.progress_apply(self.get_level, axis=1)
        # gdf2 = self.set_global_repr_point(gdf2)

        return gdf

if __name__ == '__main__':
    df = pd.DataFrame(data={'text': 'На биржевой 14 что-то произошло'}, index=[0])
    print(Geocoder().run(df=df, text_column='text'))

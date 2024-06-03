"""
This module contains classes for retrieving and working with various types of data.

@class:GeoDataGetter:
This class is used to retrieve geospatial data from OpenStreetMap (OSM) based on given OSM ID and tags.

@class:VKParser:
A class for parsing and working with VK comments and posts. Combines posts and comments into one dataframe.

@class:Streets:
A class for working with street data.

"""
import osmnx as ox
import geopandas as gpd
import pandas as pd
from sloyka.src.constants import (
    OSM_TAGS,
    GLOBAL_CRS,
    GLOBAL_METRIC_CRS,
    GLOBAL_EPSG,
)
from shapely.ops import transform
from tqdm import tqdm
import requests
import sys
import datetime
import time
import osm2geojson
import random
from typing import List, Optional
from osmapi import OsmApi

class GeoDataGetter:
    """
    This class is used to retrieve geospatial data from OpenStreetMap (OSM) based on given OSM ID and tags.

    Methods:
    - get_features_from_id: Retrieves features from the given OSM ID using the provided tags and OSM type, and returns the results as a GeoDataFrame.
    - _get_place_from_id: Retrieves the place from the given OSM ID and OSM type.
    - _process_tags: Processes the provided tags and returns a list of GeoDataFrames.
    - _get_features_from_place: Retrieves features from a specific place based on category and tag.
    - _handle_error: Handles any errors that occur during the process and prints an error message.
    """
    def get_features_from_id(
            self,
            osm_id: int,
            tags: dict,
            osm_type="R",
            selected_columns=['tag', 'element_type', 'osmid', 'name', 'geometry', 'centroid']
            ) -> gpd.GeoDataFrame:
        """
        Get features from the given OSM ID using the provided tags and OSM type, and return the results as a GeoDataFrame.
        
        Args:
            osm_id (int): The OpenStreetMap ID.
            tags (dict): The tags to filter by.
            osm_type (str, optional): The OpenStreetMap type. Defaults to "R".
            selected_columns (list, optional): The selected columns to include in the result GeoDataFrame. Defaults to ['tag', 'element_type', 'osmid', 'name', 'geometry', 'centroid'].
        
        Returns:
            gpd.GeoDataFrame: The GeoDataFrame containing the features.
        """
        place = self._get_place_from_id(osm_id, osm_type)
        gdf_list = self._process_tags(tags, place, selected_columns)

        if len(gdf_list) > 0:
            merged_gdf = (
                pd.concat(gdf_list).reset_index().loc[:, selected_columns]
            )
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)

        return merged_gdf

    def _get_place_from_id(self, osm_id, osm_type):
        place = ox.project_gdf(
            ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True)
        )
        return place

    def _process_tags(self, tags, place, selected_columns):
        gdf_list = []
        place_name = place.name.iloc[0]
        for category, category_tags in tags.items():
            for tag in tqdm(
                category_tags, desc=f"Processing category {category}"
            ):
                try:
                    gdf = self._get_features_from_place(
                        place_name, category, tag
                    )
                    gdf_list.append(gdf)
                except AttributeError:
                    self._handle_error(category, tag)
                    pass
        return gdf_list

    def _get_features_from_place(self, place_name, category, tag):
        gdf = ox.features_from_place(place_name, tags={category: tag})
        gdf.geometry.dropna(inplace=True)
        gdf["tag"] = category
        gdf["centroid"] = gdf["geometry"]

        tmpgdf = ox.projection.project_gdf(
            gdf, to_crs=GLOBAL_METRIC_CRS, to_latlong=False
        )
        tmpgdf["centroid"] = tmpgdf["geometry"].centroid
        tmpgdf = tmpgdf.to_crs(GLOBAL_CRS)
        gdf["centroid"] = tmpgdf["centroid"]
        tmpgdf = None

        return gdf

    def _handle_error(self, category, tag):
        print(
            f"\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}"
        )

class HistGeoDataGetter:
    @staticmethod
    def set_overpass_settings(date: Optional[str] = None):
        """
        Sets the overpass settings for the OpenStreetMap API.

        Parameters:
            date (Optional[str]): The date for which to retrieve data. If not provided, the current date is used.

        Returns:
            None
        """
        if date:
            ox.settings.overpass_settings = f'[out:json][timeout:600][date:"{date}"]'
        else:
            ox.settings.overpass_settings = '[out:json][timeout:600]'

    def get_features_from_id(
        self,
        osm_id: int,
        tags: dict,
        osm_type="R",
        selected_columns=['tag', 'key', 'element_type', 'osmid', 'name', 'geometry', 'centroid'],
        date: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Get features from the given OSM ID using the provided tags and OSM type, and return the results as a GeoDataFrame.

        Args:
            osm_id (int): The OpenStreetMap ID.
            tags (dict): The tags to filter by.
            osm_type (str, optional): The OpenStreetMap type. Defaults to "R".
            selected_columns (list, optional): The selected columns to include in the result GeoDataFrame. Defaults to ['tag', 'key', 'element_type', 'osmid', 'name', 'geometry', 'centroid'].
            date (Optional[str], optional): The date for which to retrieve data. If not provided, the current date is used. Defaults to None.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame containing the features.
        """
        place = HistGeoDataGetter._get_place_from_id(osm_id, osm_type)
    
        HistGeoDataGetter.set_overpass_settings(date)
        
        gdf_list = self._process_tags(tags, place, selected_columns, date)

        if len(gdf_list) > 0:
            merged_gdf = (
                pd.concat(gdf_list).reset_index().loc[:, selected_columns]
            )
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)

        if not merged_gdf.empty:
            merged_gdf = self._add_creation_timestamps(merged_gdf)

        merged_gdf = merged_gdf.dropna(subset=['name'])
        merged_gdf.reset_index(drop=True, inplace=True)
        return merged_gdf

    def _add_creation_timestamps(self, gdf):
        MyApi = OsmApi()
        timestamps = []

        for osmid in gdf['osmid']:
            try:
                object_history = MyApi.NodeHistory(osmid)
                if object_history:
                    first_version = list(object_history.values())[0]
                    timestamp = int(first_version['timestamp'].timestamp())
                    creation_timestamp = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    timestamps.append(creation_timestamp)
                else:
                    timestamps.append(None)
            except Exception as e:
                exception_type = type(e).__name__
                print(f"Error fetching timestamp for osmid {osmid} [{exception_type}]: {e}")
                timestamps.append(None)

        gdf['creation_timestamp'] = timestamps

        return gdf

    @staticmethod
    def _get_place_from_id(osm_id, osm_type):
        place = ox.project_gdf(
            ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True)
        )
        return place

    def _process_tags(self, tags, place, selected_columns, date):
        gdf_list = []
        place_name = place.name.iloc[0]
        for category, category_tags in tags.items():
            for tag in tqdm(
                category_tags, desc=f"Processing category {category}"
            ):
                try:
                    gdf = self._get_features_from_place(
                        place_name, category, tag, date
                    )
                    if len(gdf) > 0:
                        gdf_list.append(gdf)
                except Exception as e:
                    print(f"Error processing {category}-{tag}: {e}")
        return gdf_list

    def _get_features_from_place(self, place_name, category, tag, date):
        gdf = ox.features_from_place(place_name, tags={category: tag})
        gdf.geometry.dropna(inplace=True)
        gdf["tag"] = category
        gdf["centroid"] = gdf["geometry"]
        gdf["key"] = tag

        tmpgdf = ox.projection.project_gdf(
            gdf, to_crs=GLOBAL_METRIC_CRS, to_latlong=False
        )
        tmpgdf["centroid"] = tmpgdf["geometry"].centroid
        tmpgdf = tmpgdf.to_crs(GLOBAL_CRS)
        gdf["centroid"] = tmpgdf["centroid"]
        tmpgdf = None

        return gdf

    def _handle_error(self, category, tag):
        print(f"\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}")

    @staticmethod
    def get_place_name_from_osm_id(osm_id, osm_type="R"):
        place = ox.project_gdf(
            ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True)
        )
        if not place.empty:
            place_name = place.iloc[0]['display_name']
            return place_name
        else:
            return None

    @staticmethod
    def query_year_from_osm_id(osm_id, date, network_type):
        """
        Retrieves a graph from OpenStreetMap for a given OSM ID, date, and network type.

        Args:
            osm_id (int): The OpenStreetMap ID.
            date (str): The date for which to retrieve data.
            network_type (str): The network type.

        Returns:
            networkx.Graph or None: The graph from OpenStreetMap, or None if the place name is not found.

        Raises:
            None
        """
        place_name = HistGeoDataGetter.get_place_name_from_osm_id(osm_id)
        if place_name:
            HistGeoDataGetter.set_overpass_settings(date)
            G = ox.graph.graph_from_place(place_name, network_type)
            return G
        else:
            print("Place name not found.")
            return None



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
        # time.sleep(random.random())
        resp = osm2geojson.json2geojson(result)
        city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(
            Streets.global_crs
        )
        return city_bounds

class VKParser:

    API_VERISON = '5.131'
    COUNT_ITEMS = 100
    # SLEEP_TIME = 0.5
    TIMEOUT_LIMIT = 15

    @staticmethod
    def get_owner_id_by_domain(domain, access_token):
        """
    Get the owner ID of a VK group by its domain.

    Args:
        domain (str): The domain of the VK group.
        access_token (str): The access token for the VK API.

    Returns:
        int: The owner ID of the VK group, or None if the request was not successful.
    """    
        url = "https://api.vk.com/method/wall.get"
        params = {
        "domain": domain,
        "access_token": access_token,
        "v": VKParser.API_VERISON,
        }
        response = requests.get(url, params=params)
        if response.ok:
            owner_id = response.json()['response']['items'][0]['owner_id']
        else:
            owner_id = None
        return owner_id

    
    @staticmethod
    def get_group_post_ids(domain, access_token, post_num_limit, step) -> list:
        """
        A static method to retrieve a list of post IDs for a given group, based on the owner ID,
        access token, post number limit, and step size. Returns a list of post IDs.
        """
        offset = 0
        post_ids = []

        while offset < post_num_limit:
            print(offset, ' | ', post_num_limit, end="\r")
            res = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": access_token,
                    "v": VKParser.API_VERISON,
                    "domain": domain,
                    "count": step,
                    "offset": offset,
                }, timeout=10
            ).json()["response"]
            # print(res.json().keys())
            time.sleep(random.random())

            post_ids_new = [k["id"] for k in res["items"]]
            post_ids += post_ids_new
            offset += step

        return post_ids

    @staticmethod
    def get_subcomments(owner_id, post_id, access_token, params):
        """
        Retrieves subcomments from the VK API.

        Args:
            owner_id (int): The ID of the owner of the comments.
            post_id (int): The ID of the post.
            access_token (str): The access token for authentication.
            params (dict): Additional parameters for the API request.

        Returns:
            list: A list of subcomments retrieved from the API.
        """
        subcomments = []

        response = requests.get('https://api.vk.com/method/wall.getComments', params=params)
        # print(response.json().keys())
        time.sleep(random.random())
        data = response.json()

        if 'response' in data:
            for item in data['response']['items']:
                item['date'] = datetime.datetime.utcfromtimestamp(item['date']).strftime('%Y-%m-%d %H:%M:%S')
                if 'likes' in item:
                    item['likes.count'] = item['likes']['count']
                subcomments.append(item)

        return subcomments

    def get_comments(self, owner_id, post_id, access_token):
        """
        Get comments for a post on VK using the specified owner ID, post ID, and access token.
        
        Parameters:
            owner_id (int): The ID of the post owner.
            post_id (int): The ID of the post.
            access_token (str): The access token for authentication.
        
        Returns:
            list: A list of dictionaries containing comment information.
        """
        params = {
            'owner_id': owner_id,
            'post_id': post_id,
            'access_token': access_token,
            'v': VKParser.API_VERISON,
            'extended': 1,
            'count': 100,
            'need_likes': 1
        }

        comments = []

        response = requests.get('https://api.vk.com/method/wall.getComments', params=params)
        # print(response.json().keys())
        time.sleep(random.random())
        data = response.json()

        if 'response' in data:
            for item in data['response']['items']:
                if item['text'] == '':
                    continue
                item['date'] = datetime.datetime.utcfromtimestamp(item['date']).strftime('%Y-%m-%d %H:%M:%S')
                if 'likes' in item:
                    item['likes.count'] = item['likes']['count']
                comments.append(item)
                if item['thread']['count'] > 0:
                    params['comment_id'] = item['id']
                    subcomments = self.get_subcomments(owner_id, post_id, access_token, params)
                    comments.extend(subcomments)
        return comments

    @staticmethod
    def comments_to_dataframe(comments):
        """
        Convert comments to a DataFrame.

        Args:
            comments: List of comments to be converted.

        Returns:
            DataFrame: A DataFrame containing specific columns from the input comments.
        """
        df = pd.DataFrame(comments)
        df = df[['id', 'from_id', 'date', 'text', 'post_id', 'parents_stack', 'likes.count']]
        return df
    
    def run_posts(self, domain, access_token, step, cutoff_date, number_of_messages=float('inf')):
        """
        A function to retrieve posts from a social media API based on specified parameters.

        Parameters:
            owner_id (int): The ID of the owner whose posts are being retrieved.
            access_token (str): The authentication token for accessing the API.
            step (int): The number of posts to retrieve in each API call.
            cutoff_date (str): The date to stop retrieving posts (format: '%Y-%m-%d').
            number_of_messages (float): The maximum number of messages to retrieve (default is infinity).

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved posts.
        """
        token = access_token
        domain = domain
        offset = 0
        all_posts = []
        if step > number_of_messages:
            step = number_of_messages
        while offset < number_of_messages:
            print(offset, ' | ', number_of_messages, end="\r")

            response = requests.get('https://api.vk.com/method/wall.get',
                                    params={
                                        'access_token': access_token,
                                        'v': VKParser.API_VERISON,
                                        'domain': domain,
                                        'count': step,
                                        'offset': offset
                                    }
                                    )
            if response.ok:
                # print(response.json().keys())
                data = response.json()['response']['items']
                offset += step
                current_posts = pd.json_normalize(data)
                current_posts = current_posts[['date', 'id', 'text', 'views.count', 'likes.count', 'reposts.count']]
                current_posts['date'] = [datetime.datetime.fromtimestamp(current_posts['date'][i]) for i in range(len(current_posts['date']))]
                current_posts['type'] = 'post'
                all_posts.append(current_posts)
                print(current_posts.date.min())
                if any(current_posts['date'] < datetime.datetime.strptime(cutoff_date, '%Y-%m-%d')):
                    print('posts downloaded')
                    break
            else:
                continue
            time.sleep(random.random())
        df_posts = pd.concat(all_posts).reset_index(drop=True)
        df_posts = df_posts[df_posts.text.map(lambda x: len(x)) > 0]
        df_posts['text'] = df_posts['text'].str.replace(r'\n', '', regex=True)
        df_posts['link'] = df_posts['text'].str.extract(r'(https://\S+)')
        return df_posts


    def run_comments(self, domain, post_ids, access_token):

        owner_id = self.get_owner_id_by_domain(domain, access_token)
        all_comments = []
        for post_id in tqdm(post_ids):
            comments = self.get_comments(owner_id, post_id, access_token)
            all_comments.extend(comments)
        df = self.comments_to_dataframe(all_comments)
        df['type'] = 'comment'
        df = df.reset_index(drop=True)
        print('comments downloaded')
        return df
    
    def run_parser(self, domain, access_token, step, cutoff_date, number_of_messages=float('inf')):
        """
        Runs the parser with the given parameters and returns a combined DataFrame of posts and comments.
        
        :param owner_id: The owner ID for the parser.
        :param access_token: The user token for authentication.
        :param step: The step size for fetching data.
        :param cutoff_date: The cutoff date for fetching data.
        :param number_of_messages: The maximum number of messages to fetch. Defaults to positive infinity.
        :return: A combined DataFrame of posts and comments.
        """
        owner_id = self.get_owner_id_by_domain(domain, access_token)
        df_posts = self.run_posts(owner_id, access_token, step, cutoff_date, number_of_messages)
        post_ids = df_posts['id'].tolist()
        
        df_comments = self.run_comments(owner_id, post_ids, access_token)
        df_comments.loc[df_comments['parents_stack'].apply(lambda x: len(x) > 0), 'type'] = 'reply'
        for i in range(len(df_comments)):
            tmp = df_comments['parents_stack'].iloc[i]
            if tmp is not None:
                if len(tmp) > 0:
                    df_comments['parents_stack'].iloc[i] = tmp[0]
                else:
                    df_comments['parents_stack'].iloc[i] = None

        df_combined = df_comments.join(df_posts, on='post_id', rsuffix='_post')
        df_combined = pd.concat([df_posts, df_comments], ignore_index=True)
        df_group_name = self.get_group_name(domain, access_token)
        df_combined['group_name'] = df_group_name['group_name'][0]

        return df_combined
    
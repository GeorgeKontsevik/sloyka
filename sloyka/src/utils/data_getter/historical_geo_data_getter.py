import datetime
import sys
from typing import Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd
from loguru import logger
from osmapi import OsmApi
from tqdm import tqdm

from sloyka.src.utils.constants import (
    GLOBAL_CRS,
    GLOBAL_METRIC_CRS,
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
            ox.settings.overpass_settings = "[out:json][timeout:600]"

    def get_features_from_id(
        self,
        osm_id: int,
        tags: dict,
        osm_type="R",
        selected_columns=[
            "tag",
            "key",
            "element_type",
            "osmid",
            "name",
            "geometry",
            "centroid",
        ],
        date: Optional[str] = None,
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
            merged_gdf = pd.concat(gdf_list).reset_index().loc[:, selected_columns]
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)

        if not merged_gdf.empty:
            merged_gdf = self._add_creation_timestamps(merged_gdf)

        merged_gdf = merged_gdf.dropna(subset=["name"])
        merged_gdf.reset_index(drop=True, inplace=True)
        return merged_gdf

    def _add_creation_timestamps(self, gdf):
        MyApi = OsmApi()
        timestamps = []

        for osmid in gdf["osmid"]:
            try:
                object_history = MyApi.NodeHistory(osmid)
                if object_history:
                    first_version = list(object_history.values())[0]
                    timestamp = int(first_version["timestamp"].timestamp())
                    creation_timestamp = datetime.datetime.utcfromtimestamp(
                        timestamp
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    timestamps.append(creation_timestamp)
                else:
                    timestamps.append(None)
            except Exception:
                logger.exception(f"Error fetching timestamp for osmid {osmid}")
                timestamps.append(None)

        gdf["creation_timestamp"] = timestamps

        return gdf

    @staticmethod
    def _get_place_from_id(osm_id, osm_type):
        place = ox.project_gdf(ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True))
        return place

    def _process_tags(self, tags, place, selected_columns, date):
        gdf_list = []
        place_name = place.name.iloc[0]
        for category, category_tags in tags.items():
            for tag in tqdm(category_tags, desc=f"Processing category {category}"):
                try:
                    gdf = self._get_features_from_place(place_name, category, tag, date)
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
        print(
            f"\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}"
        )

    @staticmethod
    def get_place_name_from_osm_id(osm_id, osm_type="R"):
        place = ox.project_gdf(ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True))
        if not place.empty:
            place_name = place.iloc[0]["display_name"]
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
            logger.error(f"Place name not found for OSM ID {osm_id} on date {date}.")
        return None

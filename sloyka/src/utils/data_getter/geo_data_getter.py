import geopandas as gpd
import networkx as nx
import osm2geojson
import osmnx as ox
import pandas as pd
import requests
from tqdm import tqdm

from sloyka.src.utils.constants import (
    GLOBAL_CRS,
    GLOBAL_METRIC_CRS,
)


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

    @staticmethod
    def get_osm_data(osm_id: int, tags: dict) -> pd.DataFrame:
        """
        Retrieves spatial data from OSM for given tags using OSM ID and returns a DataFrame.
        """
        try:
            osm_id_rel = f"R{osm_id}"
            city_boundary_gdf = ox.geocode_to_gdf(osm_id_rel, by_osmid=True)
            polygon = city_boundary_gdf["geometry"].iloc[0]
            data = ox.features_from_polygon(polygon, tags)
            df = pd.DataFrame(data)
            df = df.dropna(subset=["name"])
            df = df.loc[:, ["name", "geometry"] + list(tags.keys())]
            return df
        except Exception as e:
            raise RuntimeError(f"Error retrieving OSM data for {osm_id_rel}: {e}")

    @staticmethod
    def get_city_bounds(osm_id: int) -> gpd.GeoDataFrame:
        """
        Method retrieves the boundary of a specified city from OSM
        using Overpass API and returns a GeoDataFrame representing
        the boundary as a polygon.
        """
        # Streets.logger.info(f"Retrieving city bounds for osm_id {osm_id}")
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
            (
            relation({osm_id});
            );
            out geom;
            """

        try:
            result = requests.get(
                overpass_url, params={"data": overpass_query}, timeout=30
            ).json()
            resp = osm2geojson.json2geojson(result)
            city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(
                GLOBAL_CRS
            )
            # Streets.logger.debug(f"City bounds retrieved: {city_bounds}")
            return city_bounds
        except requests.exceptions.RequestException as e:
            # Streets.logger.error(f"Error retrieving city bounds: {e}")
            raise e

    @staticmethod
    def get_features_from_id(
        osm_id: int,
        tags: dict,
        osm_type="R",
        selected_columns=[
            "tag",
            "element_type",
            "osmid",
            "name",
            "geometry",
            "centroid",
        ],
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
        place = GeoDataGetter._get_place_from_id(osm_id, osm_type)
        gdf_list = GeoDataGetter._process_tags(tags, place)

        if len(gdf_list) > 0:
            merged_gdf = pd.concat(gdf_list).reset_index().loc[:, selected_columns]
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)

        return merged_gdf

    @staticmethod
    def _get_place_from_id(osm_id, osm_type):
        place = ox.project_gdf(ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True))
        return place

    @staticmethod
    def _process_tags(tags, place):
        gdf_list = []
        place_name = place.name.iloc[0]
        for category, category_tags in tags.items():
            for tag in tqdm(category_tags, desc=f"Processing category {category}"):
                try:
                    gdf = GeoDataGetter._get_features_from_place(
                        place_name, category, tag
                    )
                    gdf_list.append(gdf)
                except AttributeError:
                    # GeoDataGetter._handle_error(category, tag)
                    pass
        return gdf_list

    @staticmethod
    def _get_features_from_place(place_name, category, tag):
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

    @staticmethod
    def get_drive_graph(city_bounds: gpd.GeoDataFrame) -> nx.MultiDiGraph:
        """
        Method uses the OSMnx library to retrieve the street network for a
        specified city and returns it as a NetworkX MultiDiGraph object, where
        each edge represents a street segment and each node represents
        an intersection.
        """
        # Streets.logger.info("Retrieving drive graph")
        try:
            G_drive = ox.graph_from_polygon(
                city_bounds.dissolve()["geometry"].squeeze(), network_type="drive"
            )
            # Streets.logger.debug(f"Drive graph retrieved: {G_drive}")
            return G_drive
        except Exception as e:
            # Streets.logger.error(f"Error retrieving drive graph: {e}")
            raise e

    # def _handle_error(self, category, tag):
    #     print(
    #         f"\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}"
    #     )

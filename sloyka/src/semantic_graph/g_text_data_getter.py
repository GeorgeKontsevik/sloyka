import re

import geopandas as gpd
import networkx as nx
import pandas as pd
import pymorphy3


@staticmethod
def get_tag(nodes: list, toponyms: list) -> dict:
    """
    Get attributes of part of speech for the given nodes, with the option to specify toponyms.

    Args:
        nodes (list): list of strings representing the nodes
        toponyms (list): list of strings representing the toponyms

    Returns:
        dict: dictionary containing attributes for the nodes
    """

    morph = pymorphy3.MorphAnalyzer()
    attrs = {}

    for i in nodes:
        if i not in toponyms:
            attrs[i] = str(morph.parse(i)[0].tag.POS)
        else:
            attrs[i] = "TOPONYM"

    return attrs


@staticmethod
def get_coordinates(
    G: nx.classes.graph.Graph,
    geocoded_data: gpd.GeoDataFrame,
    toponym_column: str,
    location_column: str,
    geometry_column: str,
) -> nx.classes.graph.Graph:
    """
    Get and write coordinates from geometry column in gpd.GeoDataFrame.

    Args:
        G (nx.classes.graph.Graph): Prebuild input graph.
        geocoded_data (gpd.GeoDataFrame): Data containing toponim, location and geometry of toponim.
        toponym_column (str): The name of the column containing the toponim data.
        location_column (str): The name of the column containing the location data.
        geometry_column (str): The name of the column containing the geometry data.

    Returns:
        nx.classes.graph.Graph: Graph with toponym nodes ('tag'=='TOPONYM') containing information
        about address and geometry ('Location','Lon','Lat' as node attributes)
    """
    toponyms_list = [i for i in G.nodes if G.nodes[i].get("tag") == "TOPONYM"]
    all_toponyms_list = list(geocoded_data[toponym_column])

    for i in toponyms_list:
        if i in all_toponyms_list:
            # index = all_toponyms_list.index(i)
            G.nodes[i]["Location"] = str(
                geocoded_data[location_column].iloc[all_toponyms_list.index(i)]
            )

    for i in toponyms_list:
        if i in all_toponyms_list:
            cord = geocoded_data[geometry_column].iloc[all_toponyms_list.index(i)]
            if cord is not None:
                G.nodes[i]["Lat"] = cord.x
                G.nodes[i]["Lon"] = cord.y

    return G


@staticmethod
def get_text_ids(
    G: nx.classes.graph.Graph,
    filtered_data: pd.DataFrame | gpd.GeoDataFrame,
    toponym_column: str,
    text_id_column: str,
) -> nx.classes.graph.Graph:
    """
    Update the text_ids attribute of nodes in the graph based on the provided filtered data.

    Parameters:
        G (nx.classes.graph.Graph): The input graph.
        filtered_data (pd.DataFrame or gpd.GeoDataFrame): The data to filter.
        toponym_column (str): The column name in filtered_data containing toponyms.
        text_id_column (str): The column name in filtered_data containing text IDs.

    Returns:
        nx.classes.graph.Graph: The graph with updated text_ids attributes.
    """

    toponyms_list = [i for i in G.nodes if G.nodes[i]["tag"] == "TOPONYM"]

    for i in range(len(filtered_data)):
        name = filtered_data[toponym_column].iloc[i]
        if name in toponyms_list:
            ids = [filtered_data[text_id_column].iloc[i]]

            ids = [str(k) for k in ids]

            if "text_ids" in G.nodes[name].keys():
                G.nodes[name]["text_ids"] = (
                    G.nodes[name]["text_ids"] + "," + ",".join(ids)
                )
            else:
                G.nodes[name]["text_ids"] = ",".join(ids)

    return G


@staticmethod
def get_house_text_id(
    G: nx.classes.graph.Graph,
    geocoded_data: gpd.GeoDataFrame,
    text_id_column: str,
    text_column: str,
) -> nx.classes.graph.Graph:
    """
    Get house text ids from geocoded data and assign them to the graph nodes.

    Args:
        G (nx.classes.graph.Graph): The input graph.
        geocoded_data (gpd.GeoDataFrame): Data containing geocoded information.
        text_id_column (str): The name of the column containing the text id.
        text_column (str): The name of the column containing the text.

    Returns:
        nx.classes.graph.Graph: The graph with assigned text ids to the nodes.
    """

    for i in G.nodes:
        if G.nodes[i]["tag"] == "TOPONYM":
            if re.search("\d+", i):
                id_list = G.nodes[i]["text_ids"].split(",")
                id_list = [int(j) for j in id_list]
                text = geocoded_data[text_column].loc[
                    geocoded_data[text_id_column] == id_list[0]
                ]

                G.nodes[i]["extracted_from"] = text.iloc[0]

    return G

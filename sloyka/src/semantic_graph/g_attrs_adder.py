import itertools

import geopandas as gpd
import geopy
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


@staticmethod
def add_attributes(
    G: nx.classes.graph.Graph,
    new_attributes: dict,
    attribute_tag: str,
    toponym_attributes: bool,
) -> nx.classes.graph.Graph:
    """
    Add attributes to nodes in the graph based on the specified conditions.

    Parameters:
        G (nx.classes.graph.Graph): The graph to which attributes will be added.
        new_attributes (dict): A dictionary containing the new attributes to be added.
        attribute_tag (str): The tag of the attribute to be added.
        toponym_attributes (bool): A boolean flag indicating whether to add attributes to toponyms.

    Returns:
        nx.classes.graph.Graph: The graph with the new attributes added.
    """

    if toponym_attributes:
        toponyms_list = [i for i in G.nodes if G.nodes[i].get("tag") == "TOPONYM"]
        for i in toponyms_list:
            G.nodes[i][attribute_tag] = new_attributes[i]

    else:
        word_list = [i for i in G.nodes if G.nodes[i].get("tag") != "TOPONYM"]
        for i in word_list:
            G.nodes[i][attribute_tag] = new_attributes[i]
    return G


@staticmethod
def add_city_graph(
    G: nx.classes.graph.Graph,
    districts: gpd.GeoDataFrame,
    municipals: gpd.GeoDataFrame,
    city_column: str,
    district_column: str,
    name_column: str,
    geometry_column: str,
    directed: bool = True,
) -> nx.classes.graph.Graph:
    """
    Add a city graph to the input graph based on the provided district and municipal data.

    Args:
        G (nx.classes.graph.Graph): The input graph.
        districts (gpd.GeoDataFrame): The district data.
        municipals (gpd.GeoDataFrame): The municipal data.
        city_column (str): The column name in districts containing the city data.
        district_column (str): The column name in municipals containing the district data.
        name_column (str): The column name in districts and municipals containing the name data.
        geometry_column (str): The column name in districts and municipals containing the geometry data.
        directed (bool): Whether the graph should be directed. Defaults to True.

    Returns:
        nx.classes.graph.Graph: The graph with the added city graph.
    """

    edges = []
    toponyms = [i for i in G.nodes if G.nodes[i]["tag"] == "TOPONYM"]
    city = districts[city_column].iloc[0]

    for i in range(len(districts)):
        name = districts[name_column].iloc[i]

        edges.append([city, name, "включает"])

    for i in range(len(municipals)):
        name = municipals[name_column].iloc[i]
        district = municipals[district_column].iloc[i]

        polygon = municipals[geometry_column].iloc[i]
        for j in toponyms:
            if "Lat" in G.nodes[j]:
                point = Point(G.nodes[j]["Lat"], G.nodes[j]["Lon"])

                if polygon.contains(point) or polygon.touches(point):
                    edges.append([name, j, "включает"])

        edges.append([district, name, "включает"])

    df = pd.DataFrame(edges, columns=["source", "target", "type"])

    if directed:
        city_graph = nx.from_pandas_edgelist(
            df, "source", "target", "type", create_using=nx.DiGraph
        )
    else:
        city_graph = nx.from_pandas_edgelist(df, "source", "target", "type")

    for i in range(len(districts)):
        if "population" in districts.columns:
            city_graph.nodes[districts[name_column].iloc[i]]["tag"] = "DISTRICT"
            city_graph.nodes[districts[name_column].iloc[i]]["population"] = districts[
                name_column
            ].iloc[i]
        city_graph.nodes[districts[name_column].iloc[i]][geometry_column] = str(
            districts[geometry_column].iloc[i]
        )

    for i in range(len(municipals)):
        if "population" in municipals.columns:
            city_graph.nodes[municipals[name_column].iloc[i]]["tag"] = "MUNICIPALITY"
            city_graph.nodes[municipals[name_column].iloc[i]]["population"] = (
                municipals[name_column].iloc[i]
            )
        city_graph.nodes[municipals[name_column].iloc[i]][geometry_column] = str(
            municipals[geometry_column].iloc[i]
        )

    city_graph.nodes[city]["tag"] = "CITY"

    G = nx.compose(G, city_graph)

    return G


@staticmethod
def calculate_node_distances(
    G: nx.classes.graph.Graph, directed: bool = True
) -> nx.classes.graph.Graph or nx.classes.digraph.DiGraph:
    """
    Calculate the distances between pairs of nodes in the graph and add them as edges.

    Parameters:
        G (nx.classes.graph.Graph): The input graph.
        directed (bool): Whether the graph should be directed or undirected. Defaults to True.

    Returns:
        G: NetworkX graph object with added distance edges
    """

    toponyms = [i for i in G.nodes if G.nodes[i]["tag"] == "TOPONYM"]

    combinations = list(itertools.combinations(toponyms, 2))

    distance_edges = []

    for i in tqdm(combinations):
        if "Lat" in G.nodes[i[0]] and "Lat" in G.nodes[i[1]]:
            first_point = (G.nodes[i[0]]["Lat"], G.nodes[i[0]]["Lon"])
            second_point = (G.nodes[i[1]]["Lat"], G.nodes[i[1]]["Lon"])

            distance = geopy.distance.distance(first_point, second_point).km

            distance_edges.append([i[0], i[1], "удаленность", distance])
            distance_edges.append([i[1], i[0], "удаленность", distance])

    dist_edge_df = pd.DataFrame(
        distance_edges, columns=["source", "target", "type", "distance"]
    )

    max_dist = dist_edge_df["distance"].max()
    for i in range(len(dist_edge_df)):
        dist_edge_df.at[i, "distance"] = dist_edge_df["distance"].iloc[i] / max_dist

    if directed:
        distance_graph = nx.from_pandas_edgelist(
            dist_edge_df,
            "source",
            "target",
            ["type", "distance"],
            create_using=nx.DiGraph,
        )
    else:
        distance_graph = nx.from_pandas_edgelist(
            dist_edge_df, "source", "target", ["type", "distance"]
        )

    G = nx.compose(G, distance_graph)

    return G

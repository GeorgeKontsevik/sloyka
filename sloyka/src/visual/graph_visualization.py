import random

import folium
import networkx as nx
from folium.plugins import MarkerCluster


def draw_graph_folium(graph_path, output_file=None):
    """
    Visualizes a graph from the given graph_path using Folium and MarkerCluster.

    Args:
        graph_path (str): The path to the graphml file.
        output_file (str, optional): The file to save the visualization to. Defaults to None.

    Returns:
        folium.Map: The folium map object representing the visualized graph.
    """
    G = nx.read_graphml(graph_path)

    color_mapping = {
        -1: "gray",
        0: "blue",
        1: "green",
        2: "purple",
        3: "cyan",
        4: "brown",
        5: "orange",
        6: "pink",
        7: "darkred",
        8: "yellow",
        9: "beige",
        10: "darkgreen",
        11: "lightgreen",
        12: "darkblue",
        13: "lightblue",
        14: "darkpurple",
        15: "cadetblue",
        16: "red",
        17: "lightgreen",
        18: "lightblue",
    }

    target_clusters = range(1, 19)

    m = folium.Map(
        location=[59.9343, 30.3351],
        zoom_start=10,
        tiles="cartodbdark_matter",
        control_scale=True,
    )

    for c in target_clusters:
        mc = MarkerCluster(name=f"{c} | cluster")

        for node, data in G.nodes(data=True):
            if "Lat" in data and "Lon" in data:
                main_node_location = [data["Lat"], data["Lon"]]
                for n in G.neighbors(node):
                    if "Cluster" in G.nodes[n] and G.nodes[n]["Cluster"] == c:
                        neighbor_data = G.nodes[n]
                        neighbor_location = [
                            main_node_location[1] + random.uniform(-0.0008, 0.0008),
                            main_node_location[0] + random.uniform(-0.0008, 0.0008),
                        ]
                        folium.CircleMarker(
                            location=neighbor_location,
                            radius=10,
                            color=color_mapping[G.nodes[n]["Cluster"]],
                            fill=True,
                            fill_color=color_mapping[G.nodes[n]["Cluster"]],
                            popup=neighbor_data,
                            name=f'cluster_{G.nodes[n]["Cluster"]}',
                        ).add_to(mc)
        mc.add_to(m)

    folium.LayerControl().add_to(m)
    if output_file is not None:
        m.save(output_file)

    return m

"""
@class:Semgraph:
The main class of the semantic graph module. It is aimed to build a semantic graph based on the provided data
and parameters.
More convenient to use after extracting data from geocoder.

The Semgraph class has the following methods:

@method:clean_from_dublicates:
A function to clean a DataFrame from duplicates based on specified columns.

@method:clean_from_digits:
Removes digits from the text in the specified column of the input DataFrame.

@method:clean_from_toponyms:
Clean the text in the specified text column by removing any words that match the toponyms in the name and
toponym columns.

@method:aggregate_data:
Creates a new DataFrame by aggregating the data based on the provided text and toponyms columns.
"""

import geopandas as gpd
import networkx as nx
import nltk
import pandas as pd
from transformers import BertModel, BertTokenizer  # type: ignore

from ..utils.data_preprocessing.preprocessor import (
    clean_from_digits,
    clean_from_dublicates,
    clean_from_links,
    clean_from_toponyms,
)
from .g_attrs_adder import add_attributes
from .g_text_data_getter import get_coordinates, get_tag, get_text_ids
from .keyword_extracter import extract_keywords
from .semantic_closeness_annotator import get_semantic_closeness

nltk.download("stopwords")


from sloyka.src.utils.constants import TAG_ROUTER


class Semgraph:
    """
    This is the main class of semantic graph module.
    It is aimed to build a semantic graph based on the provided data and parameters.
    More convinient to use after extracting data from geocoder.

    Param:
    bert_name: the name of the BERT model to use (default is 'DeepPavlov/rubert-base-cased')
    language: the language of the BERT model (default is 'russian')
    device: the device to use for inference (default is 'cpu')
    """

    def __init__(
        self,
        bert_name: str = "DeepPavlov/rubert-base-cased",
        language: str = "russian",
        device: str = "cpu",
    ) -> None:
        self.language = language
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.model_name = bert_name
        self.model = BertModel.from_pretrained(bert_name).to(device)

    @staticmethod
    def convert_df_to_edge_df(
        data: pd.DataFrame | gpd.GeoDataFrame,
        toponym_column: str,
        word_info_column: str = "words_score",
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        edge_list = []

        for i in data[toponym_column]:
            current_df = data.loc[data[toponym_column] == i]
            for j in range(len(current_df)):
                toponym = current_df[toponym_column].iloc[j]
                word_nodes = current_df[word_info_column].iloc[j]

                for k in word_nodes:
                    if k[2] in TAG_ROUTER.keys():
                        edge_list.append([toponym, k[0], k[1], TAG_ROUTER[k[2]]])

        edge_df = pd.DataFrame(edge_list, columns=["FROM", "TO", "distance", "type"])

        return edge_df

    def build_graph(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        id_column: str,
        text_column: str,
        text_type_column: str,
        toponym_column: str,
        toponym_name_column: str,
        toponym_type_column: str,
        post_id_column: str,
        parents_stack_column: str,
        directed: bool = True,
        location_column: str | None = None,
        geometry_column: str | None = None,
        key_score_filter: float = 0.6,
        semantic_score_filter: float = 0.75,
        top_n: int = 1,
    ) -> nx.classes.graph.Graph:
        """
        Build a graph based on the provided data.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input data to build the graph from.
            id_column (str): The column containing unique identifiers.
            text_column (str): The column containing text information.
            text_type_column (str): The column indicating the type of text.
            toponym_column (str): The column containing toponym information.
            toponym_name_column (str): The column containing toponym names.
            toponym_type_column (str): The column containing toponym types.
            post_id_column (str): The column containing post identifiers.
            parents_stack_column (str): The column containing parent-child relationships.
            directed (bool): Flag indicating if the graph is directed. Defaults to True.
            location_column (str or None): The column containing location information. Defaults to None.
            geometry_column (str or None): The column containing geometry information. Defaults to None.
            key_score_filter (float): The threshold for key score filtering. Defaults to 0.6.
            semantic_score_filter (float): The threshold for semantic score filtering. Defaults to 0.75.
            top_n (int): The number of top keywords to extract. Defaults to 1.

        Returns:
            nx.classes.graph.Graph: The constructed graph.
        """

        data = clean_from_dublicates(data, id_column)

        data = clean_from_digits(data, text_column)

        data = clean_from_toponyms(
            data, text_column, toponym_name_column, toponym_type_column
        )

        data = clean_from_links(data, text_column)

        extracted = extract_keywords(
            data,
            text_column,
            text_type_column,
            toponym_column,
            id_column,
            post_id_column,
            parents_stack_column,
            key_score_filter,
            top_n,
        )

        df = extracted[0]
        toponyms_attributes = extracted[1]
        words_attributes = extracted[2]

        preprocessed_df = self.convert_df_to_edge_df(
            data=df, toponym_column=toponym_column
        )

        words_df = get_semantic_closeness(preprocessed_df, "TO", semantic_score_filter)

        graph_df = pd.concat([preprocessed_df, words_df], ignore_index=True)
        if directed:
            G = nx.from_pandas_edgelist(
                graph_df,
                source="FROM",
                target="TO",
                edge_attr=["distance", "type"],
                create_using=nx.DiGraph(),
            )

        else:
            G = nx.from_pandas_edgelist(
                graph_df, source="FROM", target="TO", edge_attr=["distance", "type"]
            )

        nodes = list(G.nodes())
        attributes = get_tag(nodes, list(set(data[toponym_column])))

        nx.set_node_attributes(G, attributes, "tag")
        G = add_attributes(
            G=G,
            new_attributes=toponyms_attributes,
            attribute_tag="counts",
            toponym_attributes=True,
        )

        G = add_attributes(
            G=G,
            new_attributes=words_attributes,
            attribute_tag="counts",
            toponym_attributes=False,
        )

        if isinstance(data, gpd.GeoDataFrame):
            G = get_coordinates(
                G=G,
                geocoded_data=data,
                toponym_column=toponym_column,
                location_column=location_column,
                geometry_column=geometry_column,
            )

        G = get_text_ids(
            G=G,
            filtered_data=df,
            toponym_column=toponym_column,
            text_id_column=id_column,
        )

        return G

    def update_graph(
        self,
        G: nx.classes.graph.Graph,
        data: pd.DataFrame | gpd.GeoDataFrame,
        id_column: str,
        text_column: str,
        text_type_column: str,
        toponym_column: str,
        toponym_name_column: str,
        toponym_type_column: str,
        post_id_column: str,
        parents_stack_column: str,
        directed: bool = True,
        counts_attribute: str | None = None,
        location_column: str | None = None,
        geometry_column: str | None = None,
        key_score_filter: float = 0.6,
        semantic_score_filter: float = 0.75,
        top_n: int = 1,
    ) -> nx.classes.graph.Graph:
        """
        Update the input graph based on the provided data, returning the updated graph.

        Args:
            G (nx.classes.graph.Graph): The input graph to be updated.
            data (pd.DataFrame or gpd.GeoDataFrame): The input data to update the graph.
            id_column (str): The column containing unique identifiers.
            text_column (str): The column containing text information.
            text_type_column (str): The column indicating the type of text.
            toponym_column (str): The column containing toponym information.
            toponym_name_column (str): The column containing toponym names.
            toponym_type_column (str): The column containing toponym types.
            post_id_column (str): The column containing post identifiers.
            parents_stack_column (str): The column containing parent-child relationships.
            directed (bool): Flag indicating if the graph is directed. Defaults to True.
            counts_attribute (str or None): The attribute to be used for counting. Defaults to None.
            location_column (str or None): The column containing location information. Defaults to None.
            geometry_column (str or None): The column containing geometry information. Defaults to None.
            key_score_filter (float): The threshold for key score filtering. Defaults to 0.6.
            semantic_score_filter (float): The threshold for semantic score filtering. Defaults to 0.75.
            top_n (int): The number of top keywords to extract. Defaults to 1.

        Returns:
            nx.classes.graph.Graph: The updated graph.
        """

        new_G = self.build_graph(
            data,
            id_column,
            text_column,
            text_type_column,
            toponym_column,
            toponym_name_column,
            toponym_type_column,
            post_id_column,
            parents_stack_column,
            directed,
            location_column,
            geometry_column,
            key_score_filter,
            semantic_score_filter,
            top_n,
        )

        joined_G = nx.compose(G, new_G)

        if counts_attribute is not None:
            nodes = list(set(G.nodes) & set(new_G.nodes))
            for i in nodes:
                joined_G.nodes[i]["total_counts"] = (
                    G.nodes[i][counts_attribute] + new_G.nodes[i]["counts"]
                )

        return joined_G


# debugging
# if __name__ == '__main__':
#     file = open("C:\\Users\\thebe\\Downloads\\test.geojson", encoding='utf-8')
#     test_gdf = gpd.read_file(file)
#
#     sm = Semgraph()
#
#     G = sm.build_graph(test_gdf[:3000],
#                        id_column='id',
#                        text_column='text',
#                        text_type_column='type',
#                        toponym_column='only_full_street_name_numbers',
#                        toponym_name_column='initial_street',
#                        toponym_type_column='Toponims',
#                        post_id_column='post_id',
#                        parents_stack_column='parents_stack',
#                        location_column='Location',
#                        geometry_column='geometry')
#
#     # print(len(G.nodes))
#     #
#     # G = sm.update_graph(G,
#     #                     test_gdf[3000:],
#     #                     id_column='id',
#     #                     text_column='text',
#     #                     text_type_column='type',
#     #                     toponym_column='only_full_street_name',
#     #                     toponym_name_column='initial_street',
#     #                     toponym_type_column='Toponims',
#     #                     post_id_column='post_id',
#     #                     parents_stack_column='parents_stack',
#     #                     counts_attribute='counts',
#     #                     location_column='Location',
#     #                     geometry_column='geometry')
#     #
#     # print(len(G.nodes))
#     #
#     # nx.write_graphml(G, 'name.graphml', encoding='utf-8')

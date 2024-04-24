"""
@class:Semgraph:
The main class of the semantic graph module. It is aimed to build a semantic graph based on the provided data
and parameters.
More convenient to use after extracting data from geocoder.

The Semgraph class has the following methods:

@method:clean_from_duplicates:
A function to clean a DataFrame from duplicates based on specified columns.

@method:clean_from_digits:
Removes digits from the text in the specified column of the input DataFrame.

@method:clean_from_toponyms:
Clean the text in the specified text column by removing any words that match the toponyms in the name and
toponym columns.

@method:aggregate_data:
Creates a new DataFrame by aggregating the data based on the provided text and toponyms columns.
"""
import time
import json
import itertools
from tqdm.auto import tqdm
import re
import torch
import nltk
import pymorphy3
import pandas as pd
import geopandas as gpd
import networkx as nx
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from keybert import KeyBERT
import geopy.distance
from shapely.geometry import Point
from flair.data import Sentence
from flair.models import SequenceTagger

from sloyka.src.topic_modeler import TopicModeler
from sloyka.src.constants import STOPWORDS, TAG_ROUTER

nltk.download('stopwords')

RUS_STOPWORDS = stopwords.words('russian') + STOPWORDS


class Semgraph:
    """
    This is the main class of semantic graph module. 
    It is aimed to build a semantic graph based on the provided data and parameters.
    More convenient to use after extracting data from geocoder.

    Parameters:
        bert_name (str): the name of the BERT model to be used. Default is 'DeepPavlov/rubert-base-cased'
        service_model_name (str): the service model to be used, if any exists. Default is 'Glebosol/city_services'.
        language (str): the language for the BERT model, default is 'russian'
        device (str): the device to run the model on, default is 'cpu'
    """

    def __init__(self,
                 bert_name: str = 'DeepPavlov/rubert-base-cased',
                 service_model_name: str = 'Glebosol/city_services',
                 language: str = 'russian',
                 device: str = 'cpu'
                 ) -> None:

        self.language = language
        self.device = device
        self.service_model_name = service_model_name
        self.service_model = SequenceTagger.load(service_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.model_name = bert_name
        self.model = BertModel.from_pretrained(bert_name).to(device)

    @staticmethod
    def clean_from_duplicates(data: pd.DataFrame or gpd.GeoDataFrame,
                              id_column: str
                              ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        A function to clean a DataFrame from duplicates based on specified columns.
        
        Args:
            data (pd.DataFrame): The input DataFrame to be cleaned.
            id_column (str): The name of the column to use as the unique identifier.
        
        Returns:
            pd.DataFrame or gpd.GeoDataFrame: A cleaned DataFrame or GeoDataFrame without duplicates based on the
            specified text column.
        """

        uniq_df = data.drop_duplicates(subset=[id_column], keep='first')
        uniq_df = uniq_df.reset_index(drop=True)

        return uniq_df

    @staticmethod
    def clean_from_digits(data: pd.DataFrame or gpd.GeoDataFrame,
                          text_column: str
                          ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Removes digits from the text in the specified column of the input DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the text column to clean.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: The DataFrame with the specified text column cleaned from digits.
        """

        for i in range(len(data)):
            text = str(data[text_column].iloc[i]).lower()
            cleaned_text = ''.join([j for j in text if not j.isdigit()])

            data.at[i, text_column] = cleaned_text

        return data

    @staticmethod
    def clean_from_toponyms(data: pd.DataFrame or gpd.GeoDataFrame,
                            text_column: str,
                            name_column: str,
                            toponym_type_column: str
                            ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Clean the text in the specified text column by removing any words that match the toponyms in the name
        and toponym columns.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
            text_column (str): The name of the column containing the text to be cleaned.
            name_column (str): The name of the column containing the toponym name (e.g. Nevski, Moika etc.).
            toponym_type_column (str): The name of the column containing the toponym type
            (e.g. street, alley, avenue etc.).

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: The DataFrame or GeoDataFrame with the cleaned text.
        """

        for i in range(len(data)):
            text = str(data[text_column].iloc[i]).lower()
            word_list = text.split()
            toponyms = [str(data[name_column].iloc[i]).lower(), str(data[toponym_type_column].iloc[i]).lower()]

            text = ' '.join([j for j in word_list if j not in toponyms])

            data.at[i, text_column] = text

        return data

    @staticmethod
    def clean_from_links(data: pd.DataFrame or gpd.GeoDataFrame,
                         text_column: str
                         ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Clean the text in the specified text column by removing links and specific patterns.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
            text_column (str): The name of the column containing the text to be cleaned.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: The DataFrame with the cleaned text.
        """
        for i in range(len(data)):
            text = str(data[text_column].iloc[i])
            if '[id' in text and ']' in text:
                start = text.index('[')
                stop = text.index(']')

                text = text[:start] + text[stop:]

            text = re.sub(r'^https?://.*[\r\n]*', '', text, flags=re.MULTILINE)

            data.at[i, text_column] = text

        return data

    @staticmethod
    def fill_empty_toponym(data: pd.DataFrame or gpd.GeoDataFrame,
                           toponym_column: str) -> None:
        """
        A function to fill empty values in the specified toponym column with None.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input dataframe containing the toponym data.
            toponym_column (str): The name of the column containing the toponym data.

        Returns:
            None
        """

        for i in range(len(data)):
            check_raw = data[toponym_column].iloc[i]
            if check_raw == '':
                data.at[i, toponym_column] = None

        return data

    def extract_services(self,
                         data: pd.DataFrame or gpd.GeoDataFrame,
                         text_column: str,
                         ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Extracts services from the given DataFrame or GeoDataFrame and adds a 'service' column to the DataFrame
        or GeoDataFrame.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input data containing text information.
            text_column (str): The name of the column containing the text information.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: The modified data with the 'service' column added.
        """

        data["service"] = None

        print('Extracting services...')
        time.sleep(1)

        for i in tqdm(range(len(data))):
            text = data[text_column][i]
            sentence = Sentence(text)
            self.service_model.predict(sentence)
            try:
                result = sentence.to_tagged_string().split('→')
                result = result[1].lstrip()
                result = result.replace('/Service', '')
                result = json.loads(result)
                data.at[i, "service"] = result

            except:
                continue

        return data

    def extract_keywords(self,
                         data: pd.DataFrame or gpd.GeoDataFrame,
                         text_column: str,
                         semantic_key_filter: float = 0.6,
                         top_n: int = 1
                         ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Extract keywords from the given data based on certain criteria.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input data containing information.
            text_column (str): The column in the data containing text information.
            semantic_key_filter (float, optional): The threshold for semantic key filtering. Defaults to 0.75.
            top_n (int, optional): The number of top keywords to extract. Defaults to 1.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: Processed data with extracted keywords, toponym counts, and word counts.
        """

        model = KeyBERT(model=self.model)
        morph = pymorphy3.MorphAnalyzer()

        data[['word', 'key_score', 'tag']] = None

        print('Extracting keywords...')
        time.sleep(1)

        for i in tqdm(range(len(data))):
            text = data[text_column].iloc[i]
            extraction = model.extract_keywords(text, top_n=top_n, stop_words=RUS_STOPWORDS)
            if extraction:
                score = extraction[0][1]
                if score > semantic_key_filter:
                    word_score = extraction[0]
                    p = morph.parse(word_score[0])[0]
                    if p.tag.POS in TAG_ROUTER.keys():
                        word = p.normal_form
                        tag = p.tag.POS

                        word_info = (word, score, tag)

                        data.at[i, 'word'] = word
                        data.at[i, 'key_score'] = score
                        data.at[i, 'tag'] = tag

        return data

    @staticmethod
    def convert_df_to_edge_df(data: pd.DataFrame or gpd.GeoDataFrame,
                              id_column: str,
                              type_column: str,
                              post_id_column: str,
                              parents_stack_column: str,
                              toponym_column: str,
                              word_info_column: str = 'words_score'
                              ) -> pd.DataFrame or gpd.GeoDataFrame:

        types = ['post', 'comment', 'reply']

        edge_list = []

        for t in types:
            chain_gdf = data.loc[data['type'] == t].dropna(subset='only_full_street_name_numbers')
            type_ids = chain_gdf['id'].tolist()

            exclude_list = []

            for i in tqdm(type_ids, desc=f'Adding edges from {t} chains'):

                if t == 'post':
                    post_comment = data[id_column].loc[
                        (data[post_id_column] == i) & (data[toponym_column].isna())].tolist()
                    chains = data[id_column].loc[(data[post_id_column] == i) & (data[toponym_column].isna()) & (
                        data[parents_stack_column].isin(post_comment))].tolist() + post_comment + [i]

                elif t == 'comment':
                    chains = data[id_column].loc[
                                 (data[post_id_column] == i) & (~data[toponym_column].isna())].tolist() + [i]

                elif t == 'reply':
                    chains = [i]

                else:
                    continue

                toponym = data[toponym_column].loc[data[id_column] == i].iloc[0]

                for j in chains:

                    word = data['word'].loc[data[id_column] == j].iloc[0]

                    if word:
                        score = data['key_score'].loc[data[id_column] == j].iloc[0]
                        tag = data['tag'].loc[data[id_column] == j].iloc[0]
                        if tag:
                            edge_type = TAG_ROUTER[tag]
                            edge_list.append([toponym, word, score, edge_type, t])
                        else:
                            edge_list.append([toponym, word, score, None, t])

                        service = data['service'].loc[data[id_column] == j].iloc[0]
                        if service:
                            edge_list.append([word, service, None, 'описывает', t])

        edge_list_columns = ['source', 'target', 'weight', 'type']

        edge_df = pd.DataFrame(edge_list, columns=edge_list_columns)

        return edge_df

    def get_semantic_closeness(self,
                               data: pd.DataFrame or gpd.GeoDataFrame,
                               column: str,
                               similarity_filter: float = 0.75
                               ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Calculate the semantic closeness between unique words in the specified column of the input DataFrame.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
            column (str): The column in the DataFrame to calculate semantic closeness for.
            similarity_filter (float = 0.75): The score of cosine semantic proximity, from which and upper the edge
            will be generated.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: A DataFrame or GeoDataFrame containing the pairs of words with their
            similarity scores.
        """

        unic_words = tuple(set(data[column]))
        words_tokens = tuple(
            [self.tokenizer.encode(i, add_special_tokens=False, return_tensors='pt').to(self.device) for i in
             unic_words])
        potential_new_nodes_embeddings = tuple(
            [[unic_words[i], self.model(words_tokens[i]).last_hidden_state.mean(dim=1)] for i in
             range(len(unic_words))])
        new_nodes = []

        combinations = list(itertools.combinations(potential_new_nodes_embeddings, 2))

        print('Calculating semantic closeness...')
        time.sleep(1)
        for word1, word2 in tqdm(combinations):

            similarity = float(torch.nn.functional.cosine_similarity(word1[1], word2[1]))

            if similarity >= similarity_filter:
                new_nodes.append([word1[0], word2[0], similarity, 'сходство'])
                new_nodes.append([word2[0], word1[0], similarity, 'сходство'])

            time.sleep(0.001)

        result_df = pd.DataFrame(new_nodes, columns=['FROM', 'TO', 'distance', 'type'])

        return result_df

    @staticmethod
    def get_tag(nodes: list,
                toponyms: list
                ) -> dict:
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
                attrs[i] = 'TOPONYM'

        return attrs

    @staticmethod
    def get_coordinates(G: nx.classes.graph.Graph,
                        geocoded_data: gpd.GeoDataFrame,
                        toponym_column: str,
                        location_column: str,
                        geometry_column: str
                        ) -> nx.classes.graph.Graph:
        """
        Get and write coordinates from geometry column in gpd.GeoDataFrame.

        Args:
            G (nx.classes.graph.Graph): Prebuild input graph.
            geocoded_data (gpd.GeoDataFrame): Data containing toponym, location and geometry of toponym.
            toponym_column (str): The name of the column containing the toponym data.
            location_column (str): The name of the column containing the location data.
            geometry_column (str): The name of the column containing the geometry data.

        Returns:
            nx.classes.graph.Graph: Graph with toponym nodes ('tag'=='TOPONYM') containing information
            about address and geometry ('Location','Lon','Lat' as node attributes)
        """
        toponyms_list = [i for i in G.nodes if G.nodes[i].get('tag') == 'TOPONYM']
        all_toponyms_list = list(geocoded_data[toponym_column])

        for i in toponyms_list:
            if i in all_toponyms_list:
                G.nodes[i]['Location'] = str(geocoded_data[location_column].iloc[all_toponyms_list.index(i)])

        for i in toponyms_list:
            if i in all_toponyms_list:
                cord = geocoded_data[geometry_column].iloc[all_toponyms_list.index(i)]
                if cord is not None:
                    G.nodes[i]['Lat'] = cord.x
                    G.nodes[i]['Lon'] = cord.y

        return G

    @staticmethod
    def get_text_ids(G: nx.classes.graph.Graph,
                     filtered_data: pd.DataFrame or gpd.GeoDataFrame,
                     toponym_column: str,
                     text_id_column: str
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

        toponyms_list = [i for i in G.nodes if G.nodes[i]['tag'] == 'TOPONYM']

        for i in range(len(filtered_data)):
            name = filtered_data[toponym_column].iloc[i]
            if name in toponyms_list:
                ids = [filtered_data[text_id_column].iloc[i]]

                ids = [str(k) for k in ids]

                if 'text_ids' in G.nodes[name].keys():
                    G.nodes[name]['text_ids'] = G.nodes[name]['text_ids'] + ',' + ','.join(ids)
                else:
                    G.nodes[name]['text_ids'] = ','.join(ids)

        return G

    @staticmethod
    def get_house_text_id(G: nx.classes.graph.Graph,
                          geocoded_data: gpd.GeoDataFrame,
                          text_id_column: str,
                          text_column: str
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
            if G.nodes[i]['tag'] == 'TOPONYM':
                if re.search('\d+', i):
                    id_list = G.nodes[i]['text_ids'].split(',')
                    id_list = [int(j) for j in id_list]
                    text = geocoded_data[text_column].loc[geocoded_data[text_id_column] == id_list[0]]

                    G.nodes[i]['extracted_from'] = text.iloc[0]

        return G

    @staticmethod
    def add_attributes(G: nx.classes.graph.Graph,
                       new_attributes: dict,
                       attribute_tag: str,
                       toponym_attributes: bool
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
            toponyms_list = [i for i in G.nodes if G.nodes[i].get('tag') == 'TOPONYM']
            for i in toponyms_list:
                G.nodes[i][attribute_tag] = new_attributes[i]

        else:
            word_list = [i for i in G.nodes if G.nodes[i].get('tag') != 'TOPONYM']
            for i in word_list:
                G.nodes[i][attribute_tag] = new_attributes[i]
        return G

    @staticmethod
    def add_city_graph(G: nx.classes.graph.Graph,
                       districts: gpd.GeoDataFrame,
                       municipals: gpd.GeoDataFrame,
                       city_column: str,
                       district_column: str,
                       name_column: str,
                       geometry_column: str,
                       directed: bool = True
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
        toponyms = [i for i in G.nodes if G.nodes[i]['tag'] == 'TOPONYM']
        city = districts[city_column].iloc[0]

        for i in range(len(districts)):
            name = districts[name_column].iloc[i]

            edges.append([city, name, 'включает'])

        for i in range(len(municipals)):
            name = municipals[name_column].iloc[i]
            district = municipals[district_column].iloc[i]

            polygon = municipals[geometry_column].iloc[i]
            for j in toponyms:
                if 'Lat' in G.nodes[j]:
                    point = Point(G.nodes[j]['Lat'], G.nodes[j]['Lon'])

                    if polygon.contains(point) or polygon.touches(point):
                        edges.append([name, j, 'включает'])

            edges.append([district, name, 'включает'])

        df = pd.DataFrame(edges, columns=['source', 'target', 'type'])

        if directed:
            city_graph = nx.from_pandas_edgelist(df, 'source', 'target', 'type', create_using=nx.DiGraph)
        else:
            city_graph = nx.from_pandas_edgelist(df, 'source', 'target', 'type')

        for i in range(len(districts)):
            if 'population' in districts.columns:
                city_graph.nodes[districts[name_column].iloc[i]]['tag'] = 'DISTRICT'
                city_graph.nodes[districts[name_column].iloc[i]]['population'] = districts[name_column].iloc[i]
            city_graph.nodes[districts[name_column].iloc[i]][geometry_column] = str(districts[geometry_column].iloc[i])

        for i in range(len(municipals)):
            if 'population' in municipals.columns:
                city_graph.nodes[municipals[name_column].iloc[i]]['tag'] = 'MUNICIPALITY'
                city_graph.nodes[municipals[name_column].iloc[i]]['population'] = municipals[name_column].iloc[i]
            city_graph.nodes[municipals[name_column].iloc[i]][geometry_column] = str(
                municipals[geometry_column].iloc[i])

        city_graph.nodes[city]['tag'] = 'CITY'

        G = nx.compose(G, city_graph)

        return G

    @staticmethod
    def calculate_distances(G: nx.classes.graph.Graph,
                            directed: bool = True
                            ) -> nx.classes.graph.Graph or nx.classes.digraph.DiGraph:
        """
        Calculate the distances between pairs of nodes in the graph and add them as edges.

        Parameters:
            G (nx.classes.graph.Graph): The input graph.
            directed (bool): Whether the graph should be directed or undirected. Defaults to True.

        Returns:
            G: NetworkX graph object with added distance edges
        """

        toponyms = [i for i in G.nodes if G.nodes[i]['tag'] == 'TOPONYM']

        combinations = list(itertools.combinations(toponyms, 2))

        distance_edges = []

        for i in tqdm(combinations):
            if 'Lat' in G.nodes[i[0]] and 'Lat' in G.nodes[i[1]]:
                first_point = (G.nodes[i[0]]['Lat'], G.nodes[i[0]]['Lon'])
                second_point = (G.nodes[i[1]]['Lat'], G.nodes[i[1]]['Lon'])

                distance = geopy.distance.distance(first_point, second_point).km

                distance_edges.append([i[0], i[1], 'удаленность', distance])
                distance_edges.append([i[1], i[0], 'удаленность', distance])

        dist_edge_df = pd.DataFrame(distance_edges, columns=['source', 'target', 'type', 'distance'])

        max_dist = dist_edge_df['distance'].max()
        for i in range(len(dist_edge_df)):
            dist_edge_df.at[i, 'distance'] = dist_edge_df['distance'].iloc[i] / max_dist

        if directed:
            distance_graph = nx.from_pandas_edgelist(dist_edge_df,
                                                     'source',
                                                     'target',
                                                     ['type', 'distance'],
                                                     create_using=nx.DiGraph)
        else:
            distance_graph = nx.from_pandas_edgelist(dist_edge_df,
                                                     'source',
                                                     'target',
                                                     ['type', 'distance'])

        G = nx.compose(G, distance_graph)

        return G

    def build_graph(self,
                    data: pd.DataFrame or gpd.GeoDataFrame,
                    id_column: str,
                    text_column: str,
                    text_type_column: str,
                    toponym_column: str,
                    toponym_name_column: str,
                    toponym_type_column: str,
                    post_id_column: str,
                    parents_stack_column: str,
                    directed: bool = True,
                    location_column: str or None = None,
                    geometry_column: str or None = None,
                    key_score_filter: float = 0.6,
                    semantic_score_filter: float = 0.75,
                    top_n: int = 1
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

        data = self.clean_from_duplicates(data,
                                          id_column)

        data = self.clean_from_digits(data,
                                      text_column)

        data = self.clean_from_toponyms(data,
                                        text_column,
                                        toponym_name_column,
                                        toponym_type_column)

        data = self.clean_from_links(data,
                                     text_column)

        data = self.extract_services(data, text_column)

        extracted = self.extract_keywords(data,
                                          text_column,
                                          text_type_column,
                                          toponym_column,
                                          id_column,
                                          post_id_column,
                                          parents_stack_column,
                                          key_score_filter,
                                          top_n)

        df = extracted[0]
        toponyms_attributes = extracted[1]
        words_attributes = extracted[2]

        preprocessed_df = self.convert_df_to_edge_df(data=df,
                                                     toponym_column=toponym_column)

        words_df = self.get_semantic_closeness(preprocessed_df,
                                               'TO',
                                               semantic_score_filter)

        graph_df = pd.concat([preprocessed_df, words_df],
                             ignore_index=True)
        if directed:
            G = nx.from_pandas_edgelist(graph_df,
                                        source='FROM',
                                        target='TO',
                                        edge_attr=['distance', 'type'],
                                        create_using=nx.DiGraph())

        else:
            G = nx.from_pandas_edgelist(graph_df,
                                        source='FROM',
                                        target='TO',
                                        edge_attr=['distance', 'type'])

        nodes = list(G.nodes())
        attributes = self.get_tag(nodes, list(set(data[toponym_column])))

        nx.set_node_attributes(G, attributes, 'tag')
        G = self.add_attributes(G=G,
                                new_attributes=toponyms_attributes,
                                attribute_tag='counts',
                                toponym_attributes=True)

        G = self.add_attributes(G=G,
                                new_attributes=words_attributes,
                                attribute_tag='counts',
                                toponym_attributes=False)

        if type(data) is gpd.GeoDataFrame:
            G = self.get_coordinates(G=G,
                                     geocoded_data=data,
                                     toponym_column=toponym_column,
                                     location_column=location_column,
                                     geometry_column=geometry_column)

        G = self.get_text_ids(G=G,
                              filtered_data=df,
                              toponym_column=toponym_column,
                              text_id_column=id_column)

        return G

    def update_graph(self,
                     G: nx.classes.graph.Graph,
                     data: pd.DataFrame or gpd.GeoDataFrame,
                     id_column: str,
                     text_column: str,
                     text_type_column: str,
                     toponym_column: str,
                     toponym_name_column: str,
                     toponym_type_column: str,
                     post_id_column: str,
                     parents_stack_column: str,
                     directed: bool = True,
                     counts_attribute: str or None = None,
                     location_column: str or None = None,
                     geometry_column: str or None = None,
                     key_score_filter: float = 0.6,
                     semantic_score_filter: float = 0.75,
                     top_n: int = 1) -> nx.classes.graph.Graph:
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

        new_G = self.build_graph(data,
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
                                 top_n)

        joined_G = nx.compose(G, new_G)

        if counts_attribute is not None:
            nodes = list(set(G.nodes) & set(new_G.nodes))
            for i in nodes:
                joined_G.nodes[i]['total_counts'] = G.nodes[i][counts_attribute] + new_G.nodes[i]['counts']

        return joined_G


# debugging
if __name__ == '__main__':
    file = open("C:\\Users\\thebe\\Downloads\\test.geojson", encoding='utf-8')
    test_gdf = gpd.read_file(file)

    sm = Semgraph(service_model_path="C:\\Users\\thebe\\OneDrive - ITMO UNIVERSITY\\НИРМА\\models\\best-model.pt")

    service_gdf = sm.extract_services(test_gdf, 'text')

    processed_data = sm.extract_keywords(service_gdf,
                                         'text',
                                         'type',
                                         'only_full_street_name_numbers',
                                         'id',
                                         'post_id',
                                         'parents_stack',
                                         semantic_key_filter=0.75,
                                         top_n=1)

    gdf = processed_data[0].dropna(subset=['services'])
    check = gdf.iloc[0]

    print(check)
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

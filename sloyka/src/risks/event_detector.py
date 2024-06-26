"""
This module contains the EventDetection class, which is aimed to generate events and their connections based
on the application of semantic clustering method (BERTopic) on the texts in the context of an urban spatial model.

The EventDetection class has the following methods:

@method:_get_roads:
Get the road network of a city as road links and roads.

@method:_get_buildings:
Get the buildings of a city as a GeoDataFrame.

@method:_collect_population:
Collect population data for each object (building, street, link).

@method:_preprocess:
Preprocess the data.
"""

import re
from itertools import chain, combinations

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from shapely.geometry import LineString
from transformers.pipelines import pipeline
from umap import UMAP


class EventDetection:
    """
    This class is aimed to generate events and their connections.
    It is based on the application of semantic clustering method (BERTopic)
    on the texts in the context of urban spatial model
    """

    def __init__(self):
        np.random.seed(42)
        self.population_filepath = None
        self.levels = ["building", "link", "road", "global"]
        self.levels_scale = dict(zip(self.levels, list(range(2, 10, 2))))
        self.functions_weights = {
            "Безопасность": 0.12,
            "Благоустройство": 0.21,
            "Дороги": 0.18,
            "ЖКХ": 0.2,
            "Здравоохранение": 0.2,
            "Другое": 0.16,
            "Образование": 0.16,
            "Социальная защита": 0.13,
            "Строительство": 0.19,
            "Обращение с отходами": 0.19,
            "Транспорт": 0.17,
            "Экология": 0.22,
            "Энергетика": 0.19,
        }
        self.messages = None
        self.links = None
        self.buildings = None
        self.population = None
        self.topic_model = None
        self.events = None
        self.connections = None

    def _get_roads(self, city_name, city_crs) -> gpd.GeoDataFrame:
        """
        Get the road network of a city as road links and roads.
        Args:
            city_name (string): The name of the city.
            city_crs (int): The spatial reference code (CRS) of the city.
        Returns:
            links (GeoDataFrame): GeoDataFrame with the city's road links and roads.
        """
        links = ox.graph_from_place(city_name, network_type="drive")
        links = ox.utils_graph.graph_to_gdfs(links, nodes=False).to_crs(city_crs)
        links = links.reset_index(drop=True)
        links["link_id"] = links.index
        links["geometry"] = links["geometry"].buffer(7)
        links = links.to_crs(4326)
        links = links[["link_id", "name", "geometry"]]
        links.loc[links["name"].map(type) == list, "name"] = links[
            links["name"].map(type) == list
        ]["name"].map(lambda x: ", ".join(x))
        road_id_name = dict(enumerate(links.name.dropna().unique().tolist()))
        road_name_id = {v: k for k, v in road_id_name.items()}
        links["road_id"] = links["name"].replace(road_name_id)
        return links

    def _get_buildings(self) -> gpd.GeoDataFrame:
        """
        Get the buildings of a city as a GeoDataFrame
        Args:
            links(GeoDataFrame): GeoDataFrame with the city's road links and roads.
            filepath (string): The path to the GeoJSON file with building data. The default is set to 'population.geojson'.
        Returns:
            buildings (GeoDataFrame): GeoDataFrame with the city's buildings.
        """
        buildings = gpd.read_file(self.population_filepath)
        buildings = buildings[
            ["address", "building_id", "population_balanced", "geometry"]
        ]
        buildings = buildings.to_crs(4326)
        buildings["building_id"] = buildings.index
        buildings = (
            gpd.sjoin_nearest(
                buildings,
                self.links[["link_id", "road_id", "geometry"]],
                how="left",
                max_distance=500,
            )
            .drop(columns=["index_right"])
            .drop_duplicates(subset="building_id")
        )
        self.buildings = buildings
        return buildings

    def _collect_population(self) -> dict:
        """
        Collect population data for each object (building, street, link).
        """
        buildings = self.buildings.copy()
        pops_global = {0: buildings.population_balanced.sum()}
        pops_buildings = buildings["population_balanced"].to_dict()
        pops_links = (
            buildings[["population_balanced", "link_id"]]
            .groupby("link_id")
            .sum()["population_balanced"]
            .to_dict()
        )
        pops_roads = (
            buildings[["population_balanced", "road_id"]]
            .groupby("road_id")
            .sum()["population_balanced"]
            .to_dict()
        )
        pops = {
            "global": pops_global,
            "road": pops_roads,
            "link": pops_links,
            "building": pops_buildings,
        }
        self.population = pops
        return pops

    def _preprocess(self) -> gpd.GeoDataFrame:
        """
        Preprocess the data
        """
        messages = self.messages[
            [
                "Текст комментария",
                "geometry",
                "Дата и время",
                "message_id",
                "cats",
            ]
        ]
        messages = messages.sjoin(self.buildings, how="left")[
            [
                "Текст комментария",
                "address",
                "geometry",
                "building_id",
                "message_id",
                "Дата и время",
                "cats",
            ]
        ]
        messages.rename(
            columns={"Текст комментария": "text", "Дата и время": "date_time"},
            inplace=True,
        )
        messages = messages.sjoin(self.links, how="left")[
            [
                "text",
                "geometry",
                "building_id",
                "index_right",
                "name",
                "message_id",
                "date_time",
                "cats",
                "road_id",
            ]
        ]
        messages.rename(
            columns={"index_right": "link_id", "name": "road_name"},
            inplace=True,
        )
        messages = messages.join(
            self.buildings[["link_id", "road_id"]],
            on="building_id",
            rsuffix="_from_building",
        )
        messages.loc[messages.link_id.isna(), "link_id"] = messages.loc[
            messages.link_id.isna()
        ]["link_id_from_building"]
        messages.loc[messages.road_id.isna(), "road_id"] = messages.loc[
            messages.road_id.isna()
        ]["road_id_from_building"]
        messages = messages[
            [
                "message_id",
                "text",
                "geometry",
                "building_id",
                "link_id",
                "road_id",
                "date_time",
                "cats",
            ]
        ].dropna(subset="text")
        messages["cats"] = messages.cats.astype(str).str.split("; ").map(lambda x: x[0])
        messages["importance"] = messages["cats"].map(self.functions_weights)
        messages["importance"].fillna(0.16, inplace=True)
        messages["global_id"] = 0
        return messages

    def _create_model(self, min_event_size):
        """
        Create a topic model with a UMAP, HDBSCAN, and a BERTopic model.
        """
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_event_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        embedding_model = pipeline(
            "feature-extraction", model="cointegrated/rubert-tiny2"
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=hdbscan_model,
            umap_model=umap_model,
            calculate_probabilities=True,
            verbose=True,
            n_gram_range=(1, 3),
        )
        return topic_model

    def _event_from_object(
        self,
        messages,
        topic_model,
        target_column: str,
        population: dict,
        object_id: float,
        event_level: str,
    ):
        """
        Create a list of events for a given object
        (building, street, link, total).
        """
        local_messages = messages[messages[target_column] == object_id]
        message_ids = local_messages.message_id.tolist()
        docs = local_messages.text.tolist()
        if len(docs) >= 5:
            try:
                topics, probs = topic_model.fit_transform(docs)
            except TypeError:
                print("Can't reduce dimensionality or some other problem")
                return
            try:
                topics = topic_model.reduce_outliers(docs, topics)
                topic_model.update_topics(docs, topics=topics)
            except ValueError:
                print("Can't distribute all messages in topics")
            event_model = topic_model.get_topic_info()
            event_model["level"] = event_level
            event_model["object_id"] = str(object_id)
            event_model["id"] = event_model.apply(
                lambda x: f"{str(x.Topic)}_{str(x.level)}_{str(x.object_id)}",
                axis=1,
            )
            try:
                event_model["potential_population"] = population[event_level][object_id]
            except Exception:  # need to select type of error
                event_model["potential_population"] = population["global"][0]

            clustered_messages = pd.DataFrame(
                data={"id": message_ids, "text": docs, "topic_id": topics}
            )
            event_model["message_ids"] = [
                clustered_messages[clustered_messages["topic_id"] == topic][
                    "id"
                ].tolist()
                for topic in event_model.Topic
            ]
            event_model["duration"] = event_model.message_ids.map(
                lambda x: (
                    pd.to_datetime(
                        messages[messages["message_id"].isin(x)].date_time
                    ).max()
                    - pd.to_datetime(
                        messages[messages["message_id"].isin(x)].date_time
                    ).min()
                ).days
            )
            event_model["category"] = event_model.message_ids.map(
                lambda x: ", ".join(
                    messages[messages["message_id"].isin(x)].cats.mode().tolist()
                )
            )
            event_model["importance"] = event_model.message_ids.map(
                lambda x: messages[messages["message_id"].isin(x)].importance.mean()
            )
            return event_model
        else:
            return

    def _get_events(self, min_event_size) -> gpd.GeoDataFrame:
        """
        Create a list of events for all levels.
        """
        messages = self.messages.copy()
        messages_list = messages.text.tolist()
        index_list = messages.message_id.tolist()
        pops = self._collect_population()
        topic_model = self._create_model(min_event_size)
        events = [
            [
                self._event_from_object(
                    messages, topic_model, f"{level}_id", pops, oid, level
                )
                for oid in messages[f"{level}_id"].unique().tolist()
            ]
            for level in reversed(self.levels)
        ]
        events = [item for sublist in events for item in sublist if item is not None]
        events = pd.concat(list(chain(events)))
        events["geometry"] = events.message_ids.map(
            lambda x: messages[
                messages.message_id.isin(x)
            ].geometry.unary_union.representative_point()
        )
        events = gpd.GeoDataFrame(events, geometry="geometry").set_crs(4326)
        events.rename(
            columns={
                "Name": "name",
                "Representative_Docs": "docs",
                "Count": "intensity",
                "potential_population": "population",
            },
            inplace=True,
        )
        events["docs"] = events["docs"].map(
            lambda x: ", ".join(
                [str(index_list[messages_list.index(text)]) for text in x]
            )
        )
        events.message_ids = events.message_ids.map(
            lambda x: ", ".join([str(id) for id in x])
        )
        events["intensity"] = (events["intensity"] - events["intensity"].min()) / (
            events["intensity"].max() - events["intensity"].min()
        )
        events["duration"] = (events["duration"] - events["duration"].min()) / (
            events["duration"].max() - events["duration"].min()
        )
        events.loc[events.intensity == 0, "intensity"] = 0.1  # fix later
        events.loc[events.duration.isna(), "duration"] = 1  # fix later
        events["risk"] = (
            events.intensity * events.duration * events.importance * events.population
        )
        events["message_ids"] = events.message_ids.map(
            lambda x: ", ".join(list(set(x.split(", "))))
        )
        events["docs"] = events.docs.map(lambda x: ", ".join(list(set(x.split(", ")))))
        return events

    def _get_event_connections(self) -> gpd.GeoDataFrame:
        """
        Create a list of connections between events.
        """
        events = self.events.copy()
        events.index = events.id
        events.geometry = events.centroid
        weights = [
            len((set(c[0]) & set(c[1])))
            for c in combinations(self.events.message_ids, 2)
        ]
        nodes = [c for c in combinations(events.id, 2)]
        connections = pd.DataFrame(nodes, weights).reset_index()
        connections.columns = ["weight", "a", "b"]
        connections = connections[connections["weight"] > 0]
        connections = connections.join(events.geometry, on="a", rsuffix="_")
        connections = connections.join(events.geometry, on="b", rsuffix="_")
        events.reset_index(drop=True, inplace=True)
        connections["geometry"] = connections.apply(
            lambda x: LineString([x["geometry"], x["geometry_"]]), axis=1
        )
        connections.drop(columns=["geometry_"], inplace=True)
        connections = gpd.GeoDataFrame(connections, geometry="geometry").set_crs(32636)
        return connections

    def _rebalance(
        self, connections, events, levels, event_population: int, event_id: str
    ):
        """
        Rebalance the population of an event.
        """
        connections_of_event = connections[connections.a == event_id].b
        if len(connections_of_event) > 0:
            accounted_pops = events[
                events.id.isin(connections_of_event) & events.level.isin(levels)
            ].population.sum()
            if event_population >= accounted_pops:
                rebalanced_pops = event_population - accounted_pops
            else:
                connections_of_event = connections[connections.b == event_id].a
                accounted_pops = events[
                    events.id.isin(connections_of_event) & events.level.isin(levels)
                ].population.sum()
                rebalanced_pops = event_population - accounted_pops
            return rebalanced_pops
        else:
            return event_population

    def _rebalance_events(self) -> gpd.GeoDataFrame:
        """
        Rebalance the population of events.
        """
        levels = self.levels.copy()
        events = self.events.copy()
        connections = self.connections.copy()
        events_rebalanced = []
        for level in levels[1:]:
            levels_to_account = levels[: levels.index(level)]
            events_for_level = events[events.level == level]
            events_for_level["rebalanced_population"] = events_for_level.apply(
                lambda x: self._rebalance(
                    connections,
                    events,
                    levels_to_account,
                    x.population,
                    x.id,
                ),
                axis=1,
            )
            events_rebalanced.append(events_for_level)
        events_rebalanced = pd.concat(events_rebalanced)
        events_rebalanced.loc[
            events_rebalanced.rebalanced_population.isna(),
            "rebalanced_population",
        ] = 0
        events_rebalanced["population"] = events_rebalanced.rebalanced_population
        events_rebalanced.drop(columns=["rebalanced_population"], inplace=True)
        events_rebalanced.population = events_rebalanced.population.astype(int)
        events_rebalanced["population"] = (
            events_rebalanced["population"] - events_rebalanced["population"].min()
        ) / (
            events_rebalanced["population"].max()
            - events_rebalanced["population"].min()
        )
        events_rebalanced.loc[events_rebalanced.population == 0, "population"] = (
            0.01  # fix later
        )
        events_rebalanced.loc[
            events_rebalanced.population.isna()
            & events_rebalanced.level.isin(["building", "link"]),
            "population",
        ] = 0.01  # fix later
        events_rebalanced.loc[
            events_rebalanced.population.isna()
            & events_rebalanced.level.isin(["road", "global"]),
            "population",
        ] = 1  # fix later
        events_rebalanced["risk"] = (
            events_rebalanced.intensity
            * (events_rebalanced.duration + 1)
            * events_rebalanced.importance
        )
        events_rebalanced = events_rebalanced[
            ["name", "docs", "level", "id", "risk", "message_ids", "geometry"]
        ]
        return events_rebalanced

    def _filter_outliers(self):
        """
        Filter outliers.
        """
        pattern = r"^-1.*"
        events = self.events
        connections = self.connections
        print(
            len(
                events[
                    events.name.map(lambda x: True if re.match(pattern, x) else False)
                ]
            ),
            "outlier clusters of",
            len(events),
            "total clusters. Filtering...",
        )
        events = events[
            events.name.map(lambda x: False if re.match(pattern, x) else True)
        ]
        connections = connections[
            connections.a.map(lambda x: False if re.match(pattern, x) else True)
        ]
        connections = connections[
            connections.b.map(lambda x: False if re.match(pattern, x) else True)
        ]
        return events, connections

    def _prepare_messages(self):
        """
        Prepare messages for export.
        """
        messages = self.messages.copy()
        messages = messages.reset_index(drop=True)
        messages.rename(columns={"cats": "block"}, inplace=True)
        messages = messages[["message_id", "text", "geometry", "date_time", "block"]]
        messages = messages.to_crs(4326)
        return messages

    def run(
        self,
        target_texts: gpd.GeoDataFrame,
        filepath_to_population: str,
        city_name: str,
        city_crs: int,
        min_event_size: int,
    ):
        """
        Returns a GeoDataFrame of events, a GeoDataFrame of
        connections between events, and a GeoDataFrame of messages.
        """
        self.population_filepath = filepath_to_population
        self.messages = target_texts.copy()
        print("messages loaded")
        self.links = self._get_roads(city_name, city_crs)
        print("road links loaded")
        self.buildings = self._get_buildings()
        print("buildings loaded")
        self.messages = self._preprocess()
        print("messages preprocessed")
        self.events = self._get_events(min_event_size)
        print("events detected")
        self.connections = self._get_event_connections()
        print("connections generated")
        self.events = self._rebalance_events()
        print("population and risk rebalanced")
        self.events, self.connections = self._filter_outliers()
        print("outliers filtered")
        self.messages = self._prepare_messages()
        print("done!")

        return self.messages, self.events, self.connections

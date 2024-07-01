"""
This class is aimed to aggregate data by region and provide some information about it users
activity
"""

from typing import Union, Optional
import pandas as pd
import geopandas as gpd

from sloyka.src.geocoder.geocoder import Geocoder
from sloyka.src.text_classifiers import TextClassifiers
from sloyka.src.utils.data_processing.city_services_extract import City_services
from sloyka.src.risks.emotion_classifier import EmotionRecognizer

POSITIVE = ["happiness", "enthusiasm"]
NEGATIVE = ["sadness", "anger", "fear", "disgust"]


class RegionalActivity:
    """This class is aimed to produce a geodataframe with the main information about users activity.
    It uses other sloyka modules such as Geocoder, TextClassifiers, City_services and EmotionRecognizer to process data.
    Processed data is saved in class attribute 'processed_geodata' and
    after class initialization can be called with RegionalActivity.processed_geodata.


    Args:
        data (pd.DataFrame): DataFrame with posts, comments and replies in text format
        with additional information such as
        date, group_name, text_type, parents_id and so on.
        Expected to be formed from sloyka.VKParser.run class function output.
        osm_id (int): OSM ID of the place from which geograhic data should be retrieved.
        tags (dict): toponyms_dict with tags to be used in the Geocoder
        (e.g. {'admin_level': [5, 6]}).
        date (str): Date from wheach data from OSM should be retrieved.
        path_to_save (str, optional): Path to save processed geodata. Defaults to None.
        text_column (str, optional): Name of the column with text in the data. Defaults to 'text'.
        group_name_column (str, optional): Name of the column with group name in the data.
        Defaults to 'group_name'.
        repository_id (str, optional): ID of the repository to be used in TextClassifiers.
        Defaults to 'Sandrro/text_to_subfunction_v10'.
        number_of_categories (int, optional): Number of categories to be used in TextClassifiers.
        Defaults to 1.
        device (str, optional): Device type to be used in models. Defaults to 'cpu'.
        use_geocoded_data (bool, optional): Whether the input data is geocoded or not. If True skips geocoding fase. Defaults to False.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, gpd.GeoDataFrame],
        osm_id: int,
        city_tags={"place": ["town"]},
        path_to_save: Optional[str] = None,
        text_column: str = "text",
        group_name_column: str = "group_name",
        repository_id: str = "Sandrro/text_to_subfunction_v10",
        number_of_categories: int = 1,
        device: str = "cpu",
        use_geocoded_data: bool = False,
    ) -> None:
        self.data: pd.DataFrame | gpd.GeoDataFrame = data
        self.osm_id: int = osm_id
        self.city_tags = city_tags
        self.text: str = text_column
        self.group_name: str = group_name_column
        self.device: str = device
        self.path_to_save: str | None = path_to_save
        self.use_geocoded_data: bool = use_geocoded_data
        self.text_classifier = TextClassifiers(
            repository_id=repository_id,
            number_of_categories=number_of_categories,
            device_type=device,
        )
        self.processed_geodata: gpd.GeoDataFrame = self.run_sloyka_modules()
        self.top_topics = (
            self.processed_geodata.copy()["classified_text"].value_counts(
                normalize=True
            )[:5]
            * 100
        )

    def run_sloyka_modules(self) -> gpd.GeoDataFrame:
        """This function runs data with the main functions of the Geocoder, TextClassifiers,City_services and
        EmotionRecognizer classes. If path_to_save was provided it also saves data in the path.

        Returns:
            None: Data is saved in RegionalActivity.processed_geodata and written to the path if path_to_save was provided
        """

        if self.use_geocoded_data:
            processed_geodata: gpd.GeoDataFrame = self.data.copy()  # type: ignore
        else:
            processed_geodata: gpd.GeoDataFrame = Geocoder(
                device=self.device, osm_id=self.osm_id, city_tags=self.city_tags
            ).run(df=self.data, text_column=self.text, group_column=self.group_name)  # type: ignore

        processed_geodata[["classified_text", "probs"]] = (
            processed_geodata[self.text]
            .progress_map(lambda x: self.text_classifier.run_text_classifier(x))
            .to_list()
        )
        # )  # type: ignore
        processed_geodata.dropna(subset=["text"], inplace=True)
        processed_geodata = City_services().run(
            df=processed_geodata, text_column=self.text
        )
        processed_geodata["emotion_average"] = (
            EmotionRecognizer().recognize_average_emotion_from_multiple_models(df=processed_geodata, text_column=self.text)
        )  # type: ignore

        if self.path_to_save:
            processed_geodata.to_file(self.path_to_save)

        return processed_geodata

    def update_geodata(self, data: Union[pd.DataFrame, gpd.GeoDataFrame]) -> None:
        """
        Update the geodata with the given data. Will ran all stages for the new data.
        If path_to_save was provided it also saves data in the path.
        If use_geocoded_data was set to True, it will skip the geocoding stage.

        Args:
            data: The data to update the geodata with.

        Returns:
            None
        """

        self.data = data
        self.processed_geodata = self.run_sloyka_modules()

    @staticmethod
    def get_chain_ids(
        name: str,
        data: Union[pd.DataFrame, gpd.GeoDataFrame],
        id_column: str,
        name_column: str,
    ) -> list:
        """This function creates a tuple of unique identifiers of chains of posts, comments and
        replies around specified value in column.

        Args:
            name (str): value in column to select a post-comment-reply chains
            data (Union[pd.DataFrame, gpd.GeoDataFrame]): data with posts, comments and replies
            id_column (str): column with unique identifiers
            name_column (str): column to base a selection

        Returns:
            tuple: tuple of unique identifiers of chains
        """

        posts_ids = data[id_column].loc[data[name_column] == name].to_list()  # type: ignore
        comments_ids = (
            data[id_column]
            .loc[data["post_id"].isin(posts_ids) & data[name_column].isin([name, None])]
            .to_list()
        )  # type: ignore
        replies_ids = (
            data[id_column]
            .loc[
                data["parents_stack"].isin(comments_ids)
                & data[name_column].isin([name, None])
            ]
            .to_list()
        )  # type: ignore

        return tuple(sorted(list(set(posts_ids + comments_ids + replies_ids))))  # type: ignore

    @staticmethod
    def get_service_counts(
        data: pd.DataFrame, services: list, service_column: str = "City_services"
    ) -> pd.DataFrame:
        """
        Calculate the counts of each service in the given DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            services (list): The list of services to count.
            service_column (str, optional): The name of the column containing the services. Defaults to 'City_services'.

        Returns:
            pd.DataFrame: A DataFrame with two columns: 'service' and 'counts', where 'service' is the name
            of each service and 'counts' is the number of occurrences of each service in the given DataFrame.
        """

        columns = ["service", "counts"]
        res = []
        counts = 0
        for service in services:
            counts = 0
            for index_num in range(len(data)):
                if service in data[service_column].iloc[index_num]:
                    counts += 1

            res.append([service, counts])

        return pd.DataFrame(res, columns=columns)

    @staticmethod
    def get_service_ids(
        data: pd.DataFrame,
        service: str,
        service_column: str = "City_services",
        id_column: str = "id",
    ) -> list:
        """
        Get the IDs of the rows in the given DataFrame that contain the specified service.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            service (str): The service to search for.
            service_column (str, optional): The name of the column containing the services. Defaults to 'City_services'.
            id_column (str, optional): The name of the column containing the IDs. Defaults to 'id'.

        Returns:
            list: A list of IDs corresponding to the rows that contain the specified service.
        """

        res = []

        for index_num in range(len(data)):
            if service in data[service_column].iloc[index_num]:
                res.append(data[id_column].iloc[index_num])

        return res

    # TODO This function should be splited in multiple
    def get_risks(
        self,
        processed_data: Optional[gpd.GeoDataFrame] = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """This function returns a pd.DataFrame with info about social risks based on provided texts.

        Args:
            top_n (int, optional): The number of most mentioned toponyms to be calculated. Defaults to 5.

        Returns:
            pd.DataFrame: Table with info about users altitude to the service in toponyms.
        """
        if processed_data is None:
            processed_data = self.processed_geodata

        gdf_final = processed_data
        top_n_toponyms = (
            processed_data["only_full_street_name"]
            .value_counts(normalize=True)
            .index[:top_n]
        )

        columns = [
            "toponym",
            "service",
            "Part_users",
            "Part_messages",
            "positive",
            "negative",
            "neutral",
            "interpretation",
            "geometry",
        ]
        risks = []

        for i in top_n_toponyms:
            all_ids = self.get_chain_ids(
                name=i,
                data=gdf_final,
                id_column="id",
                name_column="only_full_street_name",
            )

            toponym_gdf_final = gdf_final.loc[gdf_final["id"].isin(all_ids)]

            geom = toponym_gdf_final["geometry"].dropna().iloc[0]

            part_users = len(toponym_gdf_final["from_id"].unique()) / len(
                gdf_final["from_id"].unique()
            )
            part_messages = len(toponym_gdf_final["id"].unique()) / len(
                gdf_final["id"].unique()
            )

            services = toponym_gdf_final["City_services"].apply(lambda x: list(set(x)))
            services = list(set([obj for inner_list in services for obj in inner_list]))
            services_rating = (
                self.get_service_counts(data=toponym_gdf_final, services=services)
                .sort_values(by="counts", ascending=False)[:top_n]["service"]
                .to_list()
            )

            if services:
                for service in services_rating:
                    service_ids = self.get_service_ids(
                        data=toponym_gdf_final, service=service
                    )

                    users_neg = 0
                    users_pos = 0
                    users_neu = 0

                    service_gdf = toponym_gdf_final.loc[
                        toponym_gdf_final["id"].isin(service_ids)
                    ]
                    users_id = service_gdf["from_id"].unique()

                    for user in users_id:
                        pos = 0
                        neg = 0

                        user_gdf = service_gdf.loc[service_gdf["from_id"] == user]
                        grouped = user_gdf.groupby("emotion_average")[
                            "group_name"
                        ].count()

                        neg, pos = self.count_emotions(grouped)

                        if pos > neg:
                            users_pos += 1
                        elif pos < neg:
                            users_neg += 1
                        else:
                            users_neu += 1

                    positive_coef = users_pos / len(users_id)
                    negative_coef = users_neg / len(users_id)
                    neutral_coef = users_neu / len(users_id)

                    interpretation = self.interpretate_coef(
                        negative_coef, positive_coef, neutral_coef
                    )

                    risks.append(
                        [
                            i,
                            service,
                            part_users,
                            part_messages,
                            positive_coef,
                            negative_coef,
                            neutral_coef,
                            interpretation,
                            geom,
                        ]
                    )

        risks_df = pd.DataFrame(risks, columns=columns)

        return risks_df

    @staticmethod
    def interpretate_coef(
        negative_coef: float, positive_coef: float, neutral_coef: float
    ) -> str:
        """
        Interpretate the coefficients and return a string interpretation.

        Args:
            negative_coef (float): The negative coefficient.
            positive_coef (float): The positive coefficient.
            neutral_coef (float): The neutral coefficient.

        Returns:
            str: The interpretation of the coefficients.

        Raises:
            None.

        Examples:
            >>> interpretate_coef(0.5, 0.4, 0.1)
            'keep'
            >>> interpretate_coef(0.4, 0.5, 0.1)
            'reorganize'
            >>> interpretate_coef(0.4, 0.4, 0.2)
            'controversial'
            >>> interpretate_coef(0.0, 0.0, 1.0)
            'neutral'
        """
        if negative_coef > positive_coef:
            interpretation = "reorganize"
        elif negative_coef < positive_coef:
            interpretation = "keep"
        elif (
            negative_coef == positive_coef and negative_coef != 0 and neutral_coef != 0
        ):
            interpretation = "controversial"
        else:
            interpretation = "neutral"

        return interpretation

    @staticmethod
    def count_emotions(grouped):
        """
        Counts the positive and negative emotions in the grouped data and returns the total count of negative and positive emotions.

        Args:
            grouped (pandas.DataFrame): The grouped data containing emotions.

        Returns:
            Tuple[int, int]: A tuple containing the count of negative emotions and positive emotions respectively.
        """

        pos = 0
        neg = 0

        for emotion in grouped.index:
            if emotion in POSITIVE:
                if emotion in grouped.index:
                    pos += grouped[emotion]
                else:
                    continue
            elif emotion in NEGATIVE:
                if emotion in grouped.index:
                    neg += grouped[emotion]
                else:
                    continue

        return neg, pos


if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Projects\ITMO\sloyka\sloyka\sample_data\tvoygorod34.csv",
        sep=";",
    )[:100]

    print(len(df))

    ra = RegionalActivity(data=df, osm_id=3374767, text_column="text")

    print(ra.processed_geodata["Location"], ra.processed_geodata["City_services"])
    res = gpd.GeoDataFrame(ra.get_risks(), geometry="geometry")
    res.to_file(r"C:\Projects\ITMO\sloyka\sloyka\sample_data\regional_activity.geojson")

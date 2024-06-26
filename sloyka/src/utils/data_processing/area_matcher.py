import re

from nltk.stem.snowball import SnowballStemmer
from rapidfuzz import fuzz

from sloyka.src.utils.constants import AREA_STOPWORDS, GROUP_STOPWORDS
from sloyka.src.utils.data_getter.historical_geo_data_getter import HistGeoDataGetter

stemmer = SnowballStemmer("russian")


class AreaMatcher:
    def __init__(self):
        self.area_cache = {}

    def get_df_areas(self, osm_id, tags, date):
        if osm_id not in self.area_cache:
            geo_data_getter = HistGeoDataGetter()
            df_areas = geo_data_getter.get_features_from_id(
                osm_id=osm_id, tags=tags, date=date
            )
            df_areas = df_areas[df_areas["element_type"] != "way"]
            self.area_cache[osm_id] = df_areas
        return self.area_cache[osm_id]

    def preprocess_group_name(self, group_name):
        group_name = group_name.lower()
        group_name = re.sub(r"[\"!?\u2665\u2022()|,.-:]", "", group_name)
        words_to_remove = GROUP_STOPWORDS
        for word in words_to_remove:
            group_name = re.sub(word, "", group_name, flags=re.IGNORECASE)
        return group_name

    def preprocess_area_names(self, df_areas):
        words_to_remove = AREA_STOPWORDS
        for word in words_to_remove:
            df_areas["area_name"] = df_areas["name"].str.replace(word, "", regex=True)

        df_areas["area_name_processed"] = df_areas["area_name"].str.lower()
        df_areas["area_name_processed"] = df_areas["area_name_processed"].str.replace(
            r"[\"!?\u2665\u2022()|,.-:]", "", regex=True
        )
        df_areas["area_stems"] = df_areas["area_name_processed"].apply(
            lambda x: [stemmer.stem(word) for word in x.split()]
        )
        return df_areas

    def match_group_to_area(self, group_name, df_areas):
        group_name_stems = [stemmer.stem(word) for word in group_name.split()]
        max_partial_ratio = 20
        max_token_sort_ratio = 20
        best_match = None
        admin_level = None

        for _, row in df_areas.iterrows():
            area_stems = row["area_stems"]

            partial_ratio = fuzz.partial_ratio(group_name, row["area_name_processed"])
            token_sort_ratio = fuzz.token_sort_ratio(group_name_stems, area_stems)

            if (
                partial_ratio > max_partial_ratio
                and token_sort_ratio > max_token_sort_ratio
            ):
                max_partial_ratio = partial_ratio
                max_token_sort_ratio = token_sort_ratio
                best_match = row["area_name"]
                admin_level = row["key"]

        return best_match, admin_level

    def run(self, df, osm_id, tags, date):
        df_areas = self.get_df_areas(osm_id, tags, date)
        df_areas = self.preprocess_area_names(df_areas)

        for i, group_name in enumerate(df["group_name"]):
            processed_group_name = self.preprocess_group_name(group_name)
            best_match, admin_level = self.match_group_to_area(
                processed_group_name, df_areas
            )
            df.at[i, "territory"] = best_match
            df.at[i, "admin_level"] = admin_level

        return df

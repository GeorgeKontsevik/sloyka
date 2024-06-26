from typing import List, Optional

import pandas as pd
from loguru import logger


class WordFormFinder:
    def __init__(self, osm_city_name: str):
        self.osm_city_name = osm_city_name

    def find_word_form(self, df: pd.DataFrame, strts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match free form street names extracted from social media texts to the standardized forms used in the OSM database.

        This function normalizes and finds the full street names by matching extracted street names and
        their toponyms with entries in the OSM database.

        Args:
            df (pd.DataFrame): DataFrame containing extracted street names and toponyms.
            strts_df (pd.DataFrame): DataFrame containing standardized street names and toponyms from OSM.

        Returns:
            pd.DataFrame: DataFrame with matched full street names and their variations.
        """
        results = []

        for idx, row in df.iterrows():
            result = self._process_row(row, strts_df)
            results.append(result)

        # Create a new DataFrame from results
        result_df = self._results_to_dataframe(df, results)

        return result_df

    def _process_row(self, row: pd.Series, strts_df: pd.DataFrame) -> dict:
        """
        Process a single row to find matching full street names.

        Args:
            row (pd.Series): The row to process.
            strts_df (pd.DataFrame): DataFrame containing standardized street names and toponyms from OSM.

        Returns:
            dict: A dictionary with the results for the row.
        """
        try:
            search_val = row.get("Street")
            search_toponym = row.get("Toponims")
            val_num = row.get("Numbers", "")

            if not search_val or pd.isna(search_val):
                logger.warning(f"Error processing row with street '{row.get('Street')}' and toponym '{row.get('Toponims')}")
                return {"full_street_name": None, "only_full_street_name": None}

            for col in strts_df.columns[2:]:
                matching_rows = self._find_matching_rows(
                    strts_df, col, search_val, search_toponym
                )

                if not matching_rows.empty:
                    full_streets = [
                        self._format_full_address(street, val_num)
                        for street in matching_rows["street"].values
                    ]
                    return {
                        "full_street_name": ",".join(full_streets),
                        "only_full_street_name": ",".join(
                            matching_rows["street"].values
                        ),
                    }

            # If no exact match found, check without toponym
            if search_val in strts_df[col].values:
                only_streets_full = strts_df.loc[
                    strts_df[col] == search_val, "street"
                ].values
                full_streets = [
                    self._format_full_address(street, val_num)
                    for street in only_streets_full
                ]
                return {
                    "full_street_name": ",".join(full_streets),
                    "only_full_street_name": ",".join(only_streets_full),
                }
            else:
                logger.warning(f"Error processing row with street '{row.get('Street')}' and toponym '{row.get('Toponims')}'")
                return {"full_street_name": None, "only_full_street_name": None}

        except Exception as e:
            logger.warning(
                f"Error processing row with street '{row.get('Street')}' and toponym '{row.get('Toponims')}': {e}"
            )

            return {"full_street_name": None, "only_full_street_name": None}

    def _find_matching_rows(
        self,
        strts_df: pd.DataFrame,
        col: str,
        search_val: str,
        search_toponym: Optional[str],
    ) -> pd.DataFrame:
        """
        Find rows in the OSM DataFrame that match the search value and toponym.

        Args:
            strts_df (pd.DataFrame): DataFrame containing standardized street names and toponyms from OSM.
            col (str): The column to search in.
            search_val (str): The street name to search for.
            search_toponym (Optional[str]): The toponym to match.

        Returns:
            pd.DataFrame: DataFrame with matching rows.
        """
        search_rows = strts_df.loc[strts_df[col] == search_val]

        if search_toponym:
            return search_rows[search_rows["toponim_name"] == search_toponym]
        else:
            return search_rows

    def _format_full_address(self, street: str, val_num: str) -> str:
        """
        Format the full address with the building number, city, and country.

        Args:
            street (str): The street name.
            val_num (str): Building number or additional value.

        Returns:
            str: Formatted full address.
        """
        return f"{street} {val_num} {self.osm_city_name} Россия"

    def _results_to_dataframe(
        self, df: pd.DataFrame, results: List[dict]
    ) -> pd.DataFrame:
        """
        Convert results list to a DataFrame and merge it with the original DataFrame.

        Args:
            df (pd.DataFrame): Original DataFrame.
            results (List[dict]): List of result dictionaries.

        Returns:
            pd.DataFrame: DataFrame with additional columns for full street names.
        """
        results_df = pd.DataFrame.from_records(results)
        merged_df = df.reset_index(drop=True).join(results_df)

        # Explode lists into rows and merge them back into the DataFrame
        merged_df.dropna(subset=["full_street_name", "only_full_street_name"], inplace=True)

        merged_df["location_options"] = merged_df["full_street_name"].str.split(",")
        merged_df["only_full_street_name"] = merged_df[
            "only_full_street_name"
        ].str.split(",")

        exploded_locations = (
            merged_df["location_options"].explode().rename("addr_to_geocode")
        )
        exploded_streets = (
            merged_df["only_full_street_name"].explode().rename("only_full_street_name")
        )
        final_df = pd.concat(
            [exploded_locations.to_frame(), exploded_streets.to_frame()], axis=1
        )

        merged_df.drop(columns=["only_full_street_name"], inplace=True)
        final_df = merged_df.merge(final_df, left_index=True, right_index=True)

        return final_df


# Example usage:
# extractor = AddressExtractor(osm_city_name="Москва")
# df = pd.DataFrame({"Street": ["Ленина", "Пушкина"], "Toponims": ["ул.", "пр."], "Numbers": ["12", "34"]})
# strts_df = pd.DataFrame({
#     "id": [1, 2],
#     "osm_id": [1, 2],
#     "name": ["Ленина", "Пушкина"],
#     "toponim_name": ["ул.", "пр."],
#     "street": ["улица Ленина", "проспект Пушкина"]
# })
# result = extractor.find_word_form(df, strts_df)
# print(result)

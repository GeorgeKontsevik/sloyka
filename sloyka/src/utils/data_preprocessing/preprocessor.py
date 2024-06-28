import re

import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from loguru import logger


@staticmethod
def clean_from_dublicates(data: pd.DataFrame or gpd.GeoDataFrame, id_column: str) -> pd.DataFrame or gpd.GeoDataFrame:
    """
    A function to clean a DataFrame from duplicates based on specified columns.

    Args:
        data (pd.DataFrame): The input DataFrame to be cleaned.
        id_column (str): The name of the column to use as the unique identifier.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: A cleaned DataFrame or GeoDataFrame without duplicates based on the
        specified text column.
    """

    uniq_df = data.drop_duplicates(subset=[id_column], keep="first")
    uniq_df = uniq_df.reset_index(drop=True)

    return uniq_df


@staticmethod
def clean_from_digits(data: pd.DataFrame or gpd.GeoDataFrame, text_column: str) -> pd.DataFrame or gpd.GeoDataFrame:
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
        cleaned_text = "".join([j for j in text if not j.isdigit()])

        data.at[i, text_column] = cleaned_text

    return data


@staticmethod
def clean_from_toponyms(
    data: pd.DataFrame or gpd.GeoDataFrame, text_column: str, name_column: str, toponym_type_column: str
) -> pd.DataFrame or gpd.GeoDataFrame:
    """
    Clean the text in the specified text column by removing any words that match the toponyms in the name
    and toponym columns.

    Args:
        data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to be cleaned.
        name_column (str): The name of the column containing the toponym name (e.g. Nevski, Moika etc).
        toponym_type_column (str): The name of the column containing the toponym type
        (e.g. street, alley, avenue etc.).

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: The DataFrame or GeoDataFrame with the cleaned text.
    """

    for i in range(len(data)):
        text = str(data[text_column].iloc[i]).lower()
        word_list = text.split()
        toponyms = [str(data[name_column].iloc[i]).lower(), str(data[toponym_type_column].iloc[i]).lower()]

        text = " ".join([j for j in word_list if j not in toponyms])

        data.at[i, text_column] = text

    return data


@staticmethod
def clean_from_links(data: pd.DataFrame or gpd.GeoDataFrame, text_column: str) -> pd.DataFrame or gpd.GeoDataFrame:
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
        if "[id" in text and "]" in text:
            start = text.index("[")
            stop = text.index("]")

            text = text[:start] + text[stop:]

        text = re.sub(r"^https?://.*[\r\n]*", "", text, flags=re.MULTILINE)

        data.at[i, text_column] = text

    return data


@staticmethod
def fill_empty_toponym(data: pd.DataFrame or gpd.GeoDataFrame, toponym_column: str):
    for i in range(len(data)):
        check = data[toponym_column].iloc[i]
        if check == "":
            data.at[i, toponym_column] = None

    return data

@staticmethod
def graph_to_gdf(G_drive: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Method converts the street network from a NetworkX MultiDiGraph object
    to a GeoDataFrame representing the edges (streets) with columns
    for street name, length, and geometry.
    """
    # Streets.logger.info("Converting graph to GeoDataFrame")
    try:
        gdf = ox.graph_to_gdfs(G_drive, nodes=False)
        gdf["name"].dropna(inplace=True)
        gdf = gdf[["name", "length", "geometry"]]
        gdf.reset_index(inplace=True)
        gdf = gpd.GeoDataFrame(data=gdf, geometry="geometry")
        # Streets.logger.debug(f"GeoDataFrame created: {gdf}")
        return gdf
    except Exception as e:
        # Streets.logger.error(f"Error converting graph to GeoDataFrame: {e}")
        raise e

class PreprocessorInput:
    
    @staticmethod
    def preprocess_dataframe(df, column_name) -> pd.DataFrame:
        """
        A function for preprocessing a dataframe.
        """
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].str.replace('\n', ' ').str.replace('\r', ' ')
        
        initial_row_count = len(df)
        
        df = df[df[column_name].apply(PreprocessorInput.validation_row)]
        
        processed_row_count = len(df)
        
        logger.info(f"Preprocessing is complete")
        logger.info(f"Initial df row count: {initial_row_count}")
        logger.info(f"Processed df row count: {processed_row_count}")
        # print(f"Initial df row count: {initial_row_count}")
        # print(f"Processed df row count: {processed_row_count}")
        return df
    
    @staticmethod
    def validation_row(row):
        """
        Checks whether the row matches the specified conditions.
        """
        if pd.isna(row) or row == "" or row.isspace():
            return False
        if len(row.split()) <= 1 or len(row) <= 5:
            return False
        if not re.search('[а-яА-Я]', row):
            return False
        if row.isdigit():
            return False
        return True

    @staticmethod
    def run(df, column_name='text'):
        preprocessed_df = PreprocessorInput.preprocess_dataframe(df, column_name)
        return preprocessed_df

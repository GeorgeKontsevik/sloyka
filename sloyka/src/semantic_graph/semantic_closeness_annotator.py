import itertools
import time

import geopandas as gpd
import pandas as pd
import torch
from tqdm import tqdm


def get_semantic_closeness(
    self,
    data: pd.DataFrame | gpd.GeoDataFrame,
    column: str,
    similarity_filter: float = 0.75,
) -> pd.DataFrame | gpd.GeoDataFrame:
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
        [
            self.tokenizer.encode(i, add_special_tokens=False, return_tensors="pt").to(
                self.device
            )
            for i in unic_words
        ]
    )
    potential_new_nodes_embeddings = tuple(
        [
            [unic_words[i], self.model(words_tokens[i]).last_hidden_state.mean(dim=1)]
            for i in range(len(unic_words))
        ]
    )
    new_nodes = []

    combinations = list(itertools.combinations(potential_new_nodes_embeddings, 2))

    print("Calculating semantic closeness...")
    time.sleep(1)
    for word1, word2 in tqdm(combinations):
        similarity = float(torch.nn.functional.cosine_similarity(word1[1], word2[1]))

        if similarity >= similarity_filter:
            new_nodes.append([word1[0], word2[0], similarity, "сходство"])
            new_nodes.append([word2[0], word1[0], similarity, "сходство"])

        time.sleep(0.001)

    result_df = pd.DataFrame(new_nodes, columns=["FROM", "TO", "distance", "type"])

    return result_df

"""
@class:TopicModeler:
The main class of the topic extraction module. It is aimed to dynamically model city-wide topics from texts.

The TopicModeler class has the following methods:

@method:process_topics:
The main function, which is used to process the topics for each day in the specified date range, and merge topics to create a global topic model.

@method:create_topic_model:
Generates topic model for the set of texts.

@method:handle_clusters:
Splits texts for one time period into texts in clusters and outliers.
"""
import geopandas as gpd
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from transformers.pipelines import pipeline
from loguru import logger

class TopicModeler:
    def __init__(self, gdf, start_date, end_date, min_texts=15, embedding_model_name="cointegrated/rubert-tiny2"):
        self.gdf = gdf
        self.MIN_TEXTS = min_texts
        self.gdf.text = self.gdf.text.astype(str)
        self.gdf.date = pd.to_datetime(self.gdf.date)
        self.gdf.date = self.gdf.date.dt.date
        
        self.embedding_model = pipeline("feature-extraction", model=embedding_model_name)
        self.umap_model = UMAP(n_neighbors=self.MIN_TEXTS, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        self.date_range = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='D')

    def process_topics(self):
        df_list = []
        outlier_list = []
        global_model = None

        for single_date in self.date_range:
            
            # Loop through each day in the date range
            logger.info(f"Processing date: {single_date}")
            # Filter DataFrame for the current date
            current_df = self.gdf[self.gdf['date'] == single_date.date()]
            if len(outlier_list) != 0:
                outlier_list.append(current_df)
                current_df = pd.concat(outlier_list)
                outlier_list = []
            docs = current_df.text.to_list()
            # Create clusters for the current date
            if len(current_df) >= self.MIN_TEXTS:
                current_model = BERTopic(embedding_model=self.embedding_model, umap_model=self.umap_model).fit(docs)
            else:
                outlier_list.append(current_df)
                logger.info(f"Not enough texts: {len(current_df)} for time period")
                continue
            if global_model is None:
                # Global model is current model in the beginning of the period 
                global_model = current_model
            else:
                # Global model is merged from previous and current models
                global_model = BERTopic.merge_models([global_model, current_model], min_similarity=0.9)
            
            # Create df with clusters for current model
            current_df = current_model.get_document_info(docs, current_df)
            # Filter outlier texts into separate df
            df_outliers = current_df[current_df['Topic'] == -1].drop(columns=[
                'Document', 'Topic', 'Name', 'Representation', 'Representative_Docs', 
                'Top_n_words', 'Probability', 'Representative_document'])
            outlier_list.append(df_outliers)
            current_df = current_df[current_df["Topic"] != -1]
            # Add current df without outliers to the list
            df_list.append(current_df)

        # Create final df with outliers left
        df_clusters = pd.concat((df_list + outlier_list))
        # Create final df with outliers left
        final_topics = global_model.get_topic_info()
        final_topics.index = final_topics.Name
        df_clusters.drop(columns=['Topic'], inplace=True)
        df_clusters = df_clusters.join(final_topics['Topic'], on='Name')
        df_clusters.Topic = df_clusters.Topic.fillna(-1).astype(int)
        df_clusters.drop(columns=[
            'Document', 'Representation', 'Representative_Docs', 
            'Top_n_words', 'Representative_document'], inplace=True)
        df_clusters.rename(columns={'Name':'cluster_name', 
            'Topic':'cluster_id', 'Probability':'cluster_probability'}, inplace=True)
        return df_clusters

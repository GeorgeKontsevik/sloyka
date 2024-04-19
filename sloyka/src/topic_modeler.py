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
    def __init__(self, gdf, start_date, end_date, embedding_model_name="cointegrated/rubert-tiny2"):
        self.gdf = gdf
        self.gdf.text = self.gdf.text.astype(str)
        self.gdf.date = self.gdf.date.dt.date
        
        self.embedding_model = pipeline("feature-extraction", model=embedding_model_name)
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        self.date_range = pd.date_range(start_date, end_date, freq='D')
        self.global_model = None
        self.df_outliers = None

    def process_topics(self):
        df_list = []
        outlier_list = []

        for single_date in self.date_range:
            # Loop through each day in the date range
            logger.info(f"Processing date: {single_date}")
            # Filter DataFrame for the current date
            current_df = self.filter_data_by_date(single_date.date())
            if self.df_outliers is not None:
                current_df = pd.concat([current_df, self.df_outliers])
            docs = current_df.text.to_list()
            # Create clusters for the current date
            current_topic_model = self.create_topic_model(docs)

            if single_date.date() == self.date_range[0].date():
                self.global_model = current_topic_model
            else:
                self.global_model = BERTopic.merge_models([self.global_model, current_topic_model], min_similarity=0.9)
            
            current_df, self.df_outliers = self.handle_clusters(current_df, current_topic_model, docs)
            df_list.append(current_df)

        # Create final df with outliers left
        df_list.append(self.df_outliers)
        df_clusters = pd.concat(df_list)
        # Create final df with outliers left
        final_topics = self.global_model.get_topic_info()
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

    def filter_data_by_date(self, single_date):
        return self.gdf[self.gdf['date'] == single_date]

    def create_topic_model(self, docs):
        return BERTopic(embedding_model=self.embedding_model, umap_model=self.umap_model).fit(docs)

    def handle_clusters(self, df, topic_model, docs):
        # Create df with clusters for current model
        df = topic_model.get_document_info(docs, df)
        # Filter outlier texts into separate df
        df_outliers = df[df['Topic'] == -1].drop(columns=[
            'Document', 'Topic', 'Name', 'Representation', 'Representative_Docs', 
            'Top_n_words', 'Probability', 'Representative_document'])
        df = df[df["Topic"] != -1]
        return df, df_outliers

# Example of using the class
gdf = gpd.read_file("F:/Coding/all_parsed_11_04_2024_parser_1.1_fixed.geojson")
start_date = '2024-03-01'
end_date = '2024-03-05'
topic_modeler = TopicModeler(gdf, start_date, end_date)
df_clusters = topic_modeler.process_topics()

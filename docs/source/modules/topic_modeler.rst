.. _topic_modeler:

Topic Modelling
==========================

The TopicModeler class is designed to dynamically model topics of city texts. It has a process_topics method that processes
topics for each day in the specified date range and merges topics to create a global topic model.
The clustering of posts into topics is done using the embedding models in BERTopic.
The TopicModeler.process_topics() method results in the addition of the incoming GeoDataFrame with data about
the identified topics stored. Some of the messages may not belong to any of the topic clusters, will belong to cluster -1.

@class:TopicModeler:
The main class of the topic extraction module. It is aimed to dynamically model city-wide topics from texts.

The TopicModeler class has the following methods:

@method:process_topics:
The main function, which is used to process the topics for each day in the specified date range, and merge topics to create a global topic model.


Topics = TopicModeler(gdf, start_date, end_date, min_texts=15, embedding_model_name="cointegrated/rubert-tiny2")
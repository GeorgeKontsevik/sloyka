.. _sem_graph:

Semantic graph
-------------------------------------------------

.. automodule:: sloyka.src.semantic_graph.semantic_graph_builder
   :members:
   :undoc-members:

As a result of the main method Semgraph.build_graph(), the input set of messages is cleaned from duplicates, digits, identified place names
and references. For each message, a given number of keywords is extracted using the KeyBERT library model; thanks to the application of pytorch,
the semantic proximity between keywords is determined as the cosine distance in the resulting embeddings. The final result of the module is a graph,
the nodes of which are toponyms (obtained by the geolocation module) and keywords.
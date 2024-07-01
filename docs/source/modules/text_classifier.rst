.. _text_classifier:

Text Classifier
==========================
The text_classifiers module is designed to classify texts by city functions, such as housing and utilities, public amenities, transportation,
health care, and others, using a pre-trained BERT family model in Russian. The module processes the input text and classifies it into specific urban functions using a
pre-trained rubert-tiny2 model trained on 90,000 marked accesses. The main method, run_text_classifier(), calls the model, takes text as input, and returns up to three predicted
city functions with their probability of being correctly identified.


.. automodule:: sloyka.src.risks.text_classifier
   :members:
   :undoc-members:
   :show-inheritance:
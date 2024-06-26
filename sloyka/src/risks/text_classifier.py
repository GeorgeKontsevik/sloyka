"""
This module contains the TextClassifiers class, which is aimed to classify input texts into themes or structured types of events.
It uses a Huggingface transformer model trained on rubert-tiny.
In many cases, the count of messages per theme was too low to efficiently train, so synthetic themes based
on the categories as the upper level were used (for example, 'unknown_ЖКХ').

Attributes:
- repository_id (str): The repository ID.
- number_of_categories (int): The number of categories.
- device_type (str): The type of device.

The TextClassifiers class has the following methods:

@method:initialize_classifier: Initializes the text classification pipeline with the specified model, tokenizer, and device type.

@method:run_text_classifier_topics:
 Takes a text as input and returns the predicted themes and probabilities.

@method:run_text_classifier:
 Takes a text as input and returns the predicted categories and probabilities.
"""

import pandas as pd
from transformers import pipeline

from sloyka.src.utils.exceptions import (
    ClassificationError,
    ClassifierInitializationError,
    InvalidInputError,
)


class TextClassifiers:
    def __init__(self, repository_id, number_of_categories=1, device_type=None):
        self.repository_id = repository_id
        self.number_of_categories = number_of_categories
        self.device_type = (
            device_type or -1
        )  # -1 will automatically choose the device based on availability
        self.classifier = None

    def initialize_classifier(self):
        if not self.classifier:
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model=self.repository_id,
                    tokenizer="cointegrated/rubert-tiny2",
                    device=self.device_type,
                )
            except Exception as e:
                raise ClassifierInitializationError(
                    f"Failed to initialize the classifier: {e}"
                )

    def classify_text(self, text, is_topic=False):
        if not isinstance(text, str):
            raise InvalidInputError("Input must be a string.")

        self.initialize_classifier()

        try:
            predictions = self.classifier(text, top_k=self.number_of_categories)
            preds_df = pd.DataFrame(predictions)
            categories = "; ".join(preds_df["label"].tolist())
            probabilities = "; ".join(preds_df["score"].round(3).astype(str).tolist())
        except Exception as e:
            raise ClassificationError(f"Error during text classification: {e}")

        return categories, probabilities

    def run_text_classifier_topics(self, text):
        return self.classify_text(text, is_topic=True)

    def run_text_classifier(self, text):
        return self.classify_text(text)

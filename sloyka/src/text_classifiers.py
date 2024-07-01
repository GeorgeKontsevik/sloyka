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


class TextClassifiers:
    """
    This class is aimed to classify input texts into themes, or structured types of events. It uses a Huggingface transformer model trained on rubert-tiny.
    In many cases count of messages per theme was too low to efficiently train, so we used synthetic themes based on the categories as upper level (for example, 'unknown_ЖКХ')
    """
    def __init__(
        self,
        repository_id:str,
        number_of_categories=1,
        device_type=None,
    ):
        self.REP_ID = repository_id
        self.CATS_NUM = number_of_categories
        self.device_type = device_type

    def initialize_classifier(self, device_type):
        return pipeline(
            "text-classification",
            model=self.REP_ID,
            tokenizer="cointegrated/rubert-tiny2",
            max_length=2048,
            truncation=True,
            device=device_type,
        )


    def run_text_classifier_topics(self, t):
        """
        This method takes a text as input and returns the predicted themes and probabilities.
        :param t: text to classify
        :return: list of predicted themes and probabilities
        """
        classifier = self.initialize_classifier(self.device_type)

        preds = pd.DataFrame(classifier(t, top_k=self.CATS_NUM))
        classifier.call_count = 0
        if self.CATS_NUM > 1:
            cats = "; ".join(preds["label"].tolist())
            probs = "; ".join(preds["score"].round(3).astype(str).tolist())
        else:
            cats = preds["label"][0]
            probs = preds["score"].round(3).astype(str)[0]
        return [cats, probs]
    
    def run_text_classifier(self, t):
        """
        This method takes a text as input and returns the predicted categories and probabilities.
        :param t: text to classify
        :return: list of predicted categories and probabilities
        """
        classifier = self.initialize_classifier(self.device_type)

        if isinstance(t, str):
            preds = pd.DataFrame(classifier(t, top_k=self.CATS_NUM))
            classifier.call_count = 0
            if self.CATS_NUM > 1:
                cats = "; ".join(preds["label"].tolist())
                probs = "; ".join(preds["score"].round(3).astype(str).tolist())
            else:
                cats = preds["label"][0]
                probs = preds["score"].round(3).astype(str)[0]
        else:
            print("text is not string")
            cats = None
            probs = None
        return [cats, probs]


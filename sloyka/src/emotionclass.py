"""
This module contains the EmotionClassifiers class, which is designed to categorise input texts into emotion categories.
It uses a Huggingface transformer model trained on Bert_Large by default.
The EmotionClassifiers class has the following method:

@method:recognize_emotion: Adding an emotion category.

@method:recognize_average_emotion_from_multiple_models: Adding an average emotion category or the most likely emotion 
category using multiple models.
"""

from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel
import torch
import pandas as pd
from tqdm import tqdm
import gc

class EmotionRecognizer:
    """
    This class is designed to categorise input texts into emotion categories.

        Attributes:

    - model: This attribute holds the model used for emotion recognition. It defaults to HuggingFaceModel.Text.Bert_Large, 
    but can be set to any other compatible model during the instantiation of the class.

    - device: the device to use for inference. It automatically selects 'cuda' (GPU) if a compatible GPU 
    is available and CUDA is enabled, otherwise, it falls back to 'cpu'.

    - text: The text to be analyzed.

    - df: The DataFrame containing the text to be analyzed.

    - text_column: The name of the column containing the text to be analyzed.
    """

    def __init__(self, model_name=HuggingFaceModel.Text.Bert_Large, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # Define the default model names to avoid repeated initialization
        self.default_model_names = [
            HuggingFaceModel.Text.Bert_Tiny,
            HuggingFaceModel.Text.Bert_Base,
            HuggingFaceModel.Text.Bert_Large,
            HuggingFaceModel.Text.Bert_Tiny2
        ]

    def recognize_emotion(self, text):
        """Return the emotion for a given text."""
        recognizer = TextRecognizer(model=self.model_name, device=self.device)
        emotion = recognizer.recognize(text, return_single_label=True)
        return emotion

    def recognize_average_emotion_from_multiple_models(self, df, text_column, models=None, average=True):
        """Calculate the prevailing emotion using multiple models for a DataFrame column."""
        if models is None:
            models = self.default_model_names
        else:
            # Validate that the provided models are in the default models list
            for model in models:
                if model not in self.default_model_names:
                    raise ValueError(f"Model {model} is not a valid model. Valid models are: {self.default_model_names}")

        # Initialize scores DataFrame
        scores = pd.DataFrame(0, index=df.index, columns=["happiness", "sadness", "anger", "fear", "disgust", "enthusiasm", "neutral"])

        # Process each model one by one with progress bar
        for model_name in tqdm(models, desc="Processing models"):
            try:
                print(f"Clearing cache and collecting garbage before loading model: {model_name}")
                torch.cuda.empty_cache()
                gc.collect()

                print(f"Loading model: {model_name}")
                recognizer = TextRecognizer(model=model_name, device=self.device)
                model_results = [recognizer.recognize(text, return_single_label=False) for text in df[text_column]]

                for idx, result in enumerate(model_results):
                    for emotion, score in result.items():
                        if average:
                            scores.at[df.index[idx], emotion] += score
                        else:
                            scores.at[df.index[idx], emotion] = max(scores.at[df.index[idx], emotion], score)

                # Удаление модели из памяти
                del recognizer
                torch.cuda.empty_cache()  # Очистка кеша CUDA (если используется GPU)
                gc.collect()  # Сборка мусора
                print(f"Model {model_name} processed and unloaded.")
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()

        if average:
            # Average the scores by the number of models
            scores = scores.div(len(models))

        # Determine the prevailing emotion with the highest score
        prevailing_emotions = scores.idxmax(axis=1)

        return prevailing_emotions
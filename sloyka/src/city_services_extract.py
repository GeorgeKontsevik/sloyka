import pandas as pd
import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from rapidfuzz import fuzz
from typing import List
from sloyka.src.constants import CITY_SERVICES_NAMES
tagger = SequenceTagger.load("Glebosol/city_services")

class City_services:
    def extraction_services(text):
        sentence = Sentence(text)
        tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        entity_names = [entity.text for entity in entities]
        return entity_names

    def remove_last_letter(words):
        reduced_words = [word[:-1] for word in words]
        return reduced_words

    # def replace_with_most_similar(entity_names: List[str], CITY_SERVICES_NAMES: List[str]) -> List[str]:
    #     true_city_services_names = [difflib.get_close_matches(word_entity_names, CITY_SERVICES_NAMES, n=1, cutoff=0.0)[0] for word_entity_names in entity_names]
    #     return true_city_services_names
    
    def replace_with_most_similar(list_of_entities):
        similarity_matrix = np.zeros((len(list_of_entities), len(CITY_SERVICES_NAMES)))
        for i, word1 in enumerate(list_of_entities):
            for j, word2 in enumerate(CITY_SERVICES_NAMES):
                similarity = fuzz.ratio(word1, word2) / 100.0  
                similarity_matrix[i, j] = similarity
        new_list_of_entities = list_of_entities.copy()
        for i in range(len(list_of_entities)):
            max_index = np.argmax(similarity_matrix[i])
            new_list_of_entities[i] = CITY_SERVICES_NAMES[max_index]
        return new_list_of_entities
    
    def run(self, df, text_column):
        df['City_services_extraced'] = df[text_column].apply(lambda text: City_services.extraction_services(text))
        df['City_services_cuted'] = df['City_services_extraced'].apply(lambda row: City_services.remove_last_letter(row))
        df['City_services'] = df['City_services_cuted'].apply(lambda row: City_services.replace_with_most_similar(row))
        df.drop('City_services_cuted', axis= 1 , inplace= True )
        return df
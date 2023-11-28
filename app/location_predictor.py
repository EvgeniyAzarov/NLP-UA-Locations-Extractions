import json
import re
import itertools
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline as pipeline_transformers
from optimum.pipelines import pipeline as pipeline_onnx
from lingua import Language, LanguageDetectorBuilder
from typing import List
import emoji


class LocationPredictor:
    """
    Wrapper for Location Prediction NER model.
    """

    def __init__(
        self,
        chekpoint_path_uk: str,
        chekpoint_path_ru: str,
        stoprows_path: str,
        thresh_uk: float = 0.9,
        thresh_ru: float = 0.6,
        device: str = 'cpu',
        onnx: bool = False
    ):
        self.thresh_uk = thresh_uk
        self.thresh_ru = thresh_ru

        with open(stoprows_path, 'r') as stoprows_file:
            stoprows_dict = json.load(stoprows_file)
        self.stoprows = set(itertools.chain.from_iterable(stoprows_dict.values()))

        if onnx:
            pipeline = pipeline_onnx
        else:
            pipeline = pipeline_transformers

        self.classifier_uk = pipeline(
            'token-classification',
            model=chekpoint_path_uk,
            aggregation_strategy='simple',
            device=device
        )

        self.classifier_ru = pipeline(
            'token-classification',
            model=chekpoint_path_ru,
            aggregation_strategy='simple',
            device=device
        )

        languages = [Language.UKRAINIAN, Language.RUSSIAN]
        self.lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()
    
    def _remove_stoprows(self, text):
        # must be the same as the one used for stoprows extraction
        def preprocess(string):
            string = " ".join(string.split())
            string = string.strip()
            return string 

        rows = text.split("\n")
        res_rows = [row for row in rows if preprocess(row) not in self.stoprows]
        return '\n'.join(res_rows)
    
    def _contains_uk_letters(self, input_string):
        ukrainian_letters = "їієґ"
        ukrainian_letters_uppercase = ukrainian_letters.upper()
        pattern = re.compile(f"[{ukrainian_letters}{ukrainian_letters_uppercase}]")
        return bool(pattern.search(input_string))
    def _contains_ru_letters(self, input_string):
        ru_letters = "ыъэё"
        ru_letters_uppercase = ru_letters.upper()
        pattern = re.compile(f"[{ru_letters}{ru_letters_uppercase}]")
        return bool(pattern.search(input_string))

    def _detect_lang(self, text):
        if self._contains_uk_letters(text):
            return 'uk'
        elif self._contains_ru_letters(text):
            return 'ru'
        else:
            return 'uk' if self.lang_detector.detect_language_of(text) == Language.UKRAINIAN else 'ru'
    
    def _contains_emoji(self, text):
        for char in text:
            if emoji.is_emoji(char):
                return True
        else:
            return False
        
    def _extract_locations(self, text):
        text = self._remove_stoprows(text)
        lang = self._detect_lang(text)

        classifier = None
        threshold = None

        if lang == 'uk':
            classifier = self.classifier_uk
            threshold = self.thresh_uk
        else:
            classifier = self.classifier_ru
            threshold = self.thresh_ru

        ents = classifier(text)
        locs = []
        for ent in ents:
            if ent['score'] >= threshold and "#" not in ent['word'] and not self._contains_emoji(ent['word']):
                word = text[ent['start']:ent['end']].split('\n')[0]
                locs.append(word)

        return locs 
    
    def _postprocess(self, texts, ents, threshold):
        locations = []
        for i, text in enumerate(texts):
            locs = []
            for ent in ents[i]:
                if ent['score'] >= threshold \
                    and "#" not in ent['word'] \
                    and not self._contains_emoji(ent['word']):
                    word = text[ent['start']:ent['end']].split('\n')[0]
                    locs.append(word)
            locations.append(locs)

        return locations
    
    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Process texts input.

        Args:
            texts (List[str]): input texts.
            verbose (bool): show logging during inference 
        Returns:
            List[List[str]]: list of lists of locations.
        """
        df = pd.DataFrame(texts, columns=['text'])
        df['text'] = df['text'].apply(self._remove_stoprows)
        df['lang'] = df['text'].apply(self._detect_lang)
        df_uk = df[df['lang'] == 'uk']
        df_ru = df[df['lang'] == 'ru']

        locs_uk, locs_ru = [], []

        if len(df_uk) > 0:
            ents_uk = self.classifier_uk(df_uk['text'].to_list())
            locs_uk = self._postprocess(df_uk['text'].to_list(), ents_uk, self.thresh_uk)

        if len(df_ru) > 0:
            ents_ru = self.classifier_ru(df_ru['text'].to_list())
            locs_ru = self._postprocess(df_ru['text'].to_list(), ents_ru, self.thresh_ru)
        
        df.loc[df['lang'] == 'uk', "locations"] = pd.Series(locs_uk).values
        df.loc[df['lang'] == 'ru', "locations"] = pd.Series(locs_ru).values

        return df['locations'].to_list()

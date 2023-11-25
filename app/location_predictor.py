import json
import re
import itertools
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, pipeline
from lingua import Language, LanguageDetectorBuilder
from typing import List


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
    ):
        self.thresh_uk = thresh_uk
        self.thresh_ru = thresh_ru

        with open(stoprows_path, 'r') as stoprows_file:
            stoprows_dict = json.load(stoprows_file)
        self.stoprows = set(itertools.chain.from_iterable(stoprows_dict.values()))

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
            if ent['score'] >= threshold and "#" not in ent['word']:
                word = text[ent['start']:ent['end']].split('\n')[0]
                locs.append(word)

        return locs 
    
    def predict(self, texts: List[str], verbose=False) -> List[List[str]]:
        """
        Process texts input.

        Args:
            texts (List[str]): input texts.
            verbose (bool): show logging during inference 
        Returns:
            List[List[str]]: list of lists of locations.
        """

        locations = []
        for text in tqdm(texts, disable=(not verbose)):
            locations.append(self._extract_locations(text))

        return locations
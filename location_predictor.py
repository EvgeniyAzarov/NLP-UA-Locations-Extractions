import json
import itertools
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, pipeline
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
        thresh_ru: float = 0.8,
        device: str = 'cpu',
    ):
        self.thresh_uk = thresh_uk
        self.thresh_ru = thresh_ru

        with open(stoprows_path, 'r') as stoprows_file:
            stoprows_dict = json.load(stoprows_file)
        self.stoprows = set(itertools.chain.from_iterable(stoprows_dict.values()))
    
    def _remove_stoprows(self, text):
        # must be the same as the one used for stoprows extraction
        def preprocess(string):
            string = " ".join(string.split())
            string = string.strip()
            return string 

        rows = text.split("\n")
        res_rows = [row for row in rows if preprocess(row) not in self.stoprows]
        return '\n'.join(res_rows)
        
    def _extract_locations(self, text):
        text = self._remove_stoprows(text)


    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Process texts input.

        Args:
            texts (List[str]): input texts.
        Returns:
            List[List[str]]: list of lists of locations.
        """

        locations = []
        for text in tqdm(texts):
            locations.append(self._extract_locations(text))

        return locations
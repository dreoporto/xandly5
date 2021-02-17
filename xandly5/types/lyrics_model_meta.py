
from typing import Optional
from tensorflow import keras
from xandly5.ai_ml_model.catalog import Catalog


class LyricsModelMeta:

    """
    stores the model and related data:
        - model_file - h5 model file name
        - lyrics_file - lyrics text file name
        - model - keras/tensorflow model
        - catalog associated with this model
    """

    def __init__(self, model_file: str, lyrics_file: str):
        self.model_file = model_file
        self.lyrics_file = lyrics_file
        self.model: Optional[keras.Sequential] = None
        self.catalog: Optional[Catalog] = None

    def generate_lyrics_text(self, seed_text: str, word_count: int) -> str:
        """
        generate lyrics using the specific catalog associated with this model

        :param seed_text: starting text
        :param word_count: number of words to generate
        :return: seed + generated text
        """
        return self.catalog.generate_lyrics_text(self.model, seed_text, word_count)

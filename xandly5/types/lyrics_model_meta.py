
from typing import Optional
from tensorflow import keras
from xandly5.ai_ml_model.catalog import Catalog


class LyricsModelMeta:

    def __init__(self, model_file: str, lyrics_file: str):
        self.model_file = model_file
        self.lyrics_file = lyrics_file
        self.model: Optional[keras.Sequential] = None
        self.catalog: Optional[Catalog] = None


from typing import Dict, Optional
from enum import IntEnum
from tensorflow import keras

from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from dre_lib.dre_time import Stopwatch


class LyricsModelEnum(IntEnum):
    IRISH_LIT = 1
    CLASSIC_FOLK = 2


class LyricsModelWithInfo:

    def __init__(self, model_file: str, lyrics_file: str):
        self.model_file = model_file
        self.lyrics_file = lyrics_file
        self.model: Optional[keras.Sequential] = None


def _load_lyrics_models() -> Dict[LyricsModelEnum, LyricsModelWithInfo]:
    print('loading lyrics models')
    models: Dict[LyricsModelEnum, LyricsModelWithInfo] = {
        LyricsModelEnum.IRISH_LIT: LyricsModelWithInfo('nlp_irish_lit.h5', 'irish-lyrics-eof.txt')
    }

    for item, value in models.items():
        value.model = keras.models.load_model(f'../ai_ml_model/{value.model_file}')

    return models


# load using global variable
keras.backend.clear_session()
_lyrics_models = _load_lyrics_models()


class LyricsGenerator:

    _PADDING: str = 'pre'

    def __init__(self):
        print('init LyricsGenerator instance')
        # nothing to do here yet

    def generate_lyrics(self, model_id: LyricsModelEnum, seed_text: str,
                        word_group_count: int, words_to_generate: int) -> str:

        model_with_info = _lyrics_models[model_id]
        catalog = Catalog()
        catalog.add_file_to_catalog(f'../ai_ml_model/lyrics_files/{model_with_info.lyrics_file}')
        max_sequence_length, total_words, features, labels = catalog.tokenize_catalog(self._PADDING)

        lyrics_text = catalog.generate_lyrics_text(
            model_with_info.model,
            seed_text=seed_text,
            padding=self._PADDING,
            word_count=words_to_generate,
            max_sequence_length=max_sequence_length
        )

        lyrics_text = LyricsFormatter.format_lyrics(lyrics_text, word_group_count=word_group_count)

        return lyrics_text


def main():
    stopwatch = Stopwatch()
    stopwatch.start()
    generator = LyricsGenerator()
    lyrics = generator.generate_lyrics(LyricsModelEnum.IRISH_LIT, 'i wish to see green fields once more',
                                       word_group_count=4, words_to_generate=96)
    print(lyrics)
    stopwatch.stop()


if __name__ == '__main__':
    main()

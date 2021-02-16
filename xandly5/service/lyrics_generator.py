
from typing import Dict, Optional
from enum import IntEnum
from tensorflow import keras

from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from ptmlib.time import Stopwatch


class LyricsModelEnum(IntEnum):
    IRISH_LIT = 1
    SONNETS = 2


class LyricsModel:

    def __init__(self, model_file: str, lyrics_file: str, is_delimited: bool = False):
        self.model_file = model_file
        self.lyrics_file = lyrics_file
        self.is_delimited = is_delimited
        self.model: Optional[keras.Sequential] = None
        self.catalog: Optional[Catalog] = None


def _load_lyrics_models() -> Dict[LyricsModelEnum, LyricsModel]:
    print('loading lyrics models')
    models: Dict[LyricsModelEnum, LyricsModel] = {
        LyricsModelEnum.IRISH_LIT: LyricsModel('nlp_irish_lit.h5', 'irish-lyrics-eof.txt'),
        LyricsModelEnum.SONNETS: LyricsModel('shakespeare_sonnet.h5', 'shakespeare-sonnets-data.txt',
                                             is_delimited=True),
    }

    for _, lyrics_model in models.items():
        lyrics_model.model = keras.models.load_model(f'../ai_ml_model/saved_models/{lyrics_model.model_file}')
        lyrics_model.catalog = Catalog()

        if lyrics_model.is_delimited:
            lyrics_model.catalog.add_csv_file_to_catalog(f'../ai_ml_model/lyrics_files/{lyrics_model.lyrics_file}',
                                                         text_column=0, delimiter='\t')
        else:
            lyrics_model.catalog.add_file_to_catalog(f'../ai_ml_model/lyrics_files/{lyrics_model.lyrics_file}')

        lyrics_model.catalog.tokenize_catalog()

    return models


# load using global variable
keras.backend.clear_session()
_lyrics_models = _load_lyrics_models()


class LyricsGenerator:

    def __init__(self, model_id: LyricsModelEnum):
        print('init LyricsGenerator instance')
        self.model_with_info = _lyrics_models[model_id]

    def generate_lyrics(self, seed_text: str, word_group_count: int, words_to_generate: int) -> str:

        lyrics_text = self.model_with_info.catalog.generate_lyrics_text(
            self.model_with_info.model,
            seed_text=seed_text,
            word_count=words_to_generate
        )

        lyrics_text = LyricsFormatter.format_lyrics(lyrics_text, word_group_count=word_group_count)

        return lyrics_text


def main():
    stopwatch = Stopwatch()

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.IRISH_LIT)
    lyrics = generator.generate_lyrics('i wish to see green fields once more',
                                       word_group_count=4, words_to_generate=96)
    print(lyrics)
    stopwatch.stop()

    # lyrics = 'what lady is that which doth enrich the hand of yonder knight'
    lyrics = 'you\'re my only hope'

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.SONNETS)
    lyrics = generator.generate_lyrics(lyrics, word_group_count=8, words_to_generate=92)
    print(lyrics)
    stopwatch.stop()


if __name__ == '__main__':
    main()

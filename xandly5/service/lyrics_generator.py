from typing import Dict
from tensorflow import keras
from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from xandly5.types.lyrics_model_meta import LyricsModelMeta
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from ptmlib.time import Stopwatch


def _load_lyrics_models() -> Dict[LyricsModelEnum, LyricsModelMeta]:
    print('loading lyrics models')
    models: Dict[LyricsModelEnum, LyricsModelMeta] = {
        LyricsModelEnum.SONNETS: LyricsModelMeta('shakespeare_sonnet.h5', 'shakespeare-sonnets-lyrics.txt'),
        LyricsModelEnum.POE_POEM: LyricsModelMeta('poe_poem.h5', 'poe-poem-lines.txt')
    }

    for _, lyrics_model in models.items():
        lyrics_model.model = keras.models.load_model(f'../ai_ml_model/saved_models/{lyrics_model.model_file}')
        lyrics_model.catalog = Catalog()
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


# TODO AEO MOVE THIS OUT
def speak_lyrics(lyrics: str):
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    engine.setProperty('rate', 160)
    # noinspection PyUnresolvedReferences
    engine.setProperty('voice', voices[1].id)

    engine.say(lyrics)

    engine.runAndWait()
    engine.stop()


def main():
    stopwatch = Stopwatch()

    lyrics = 'evening fountians lit loss'

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.SONNETS)
    lyrics = generator.generate_lyrics(lyrics, word_group_count=8, words_to_generate=92)
    print(lyrics)
    stopwatch.stop()

    speak_lyrics(lyrics)

    lyrics = 'deep sleep lights the oasis'

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.POE_POEM)
    lyrics = generator.generate_lyrics(lyrics, word_group_count=5, words_to_generate=95)
    print(lyrics)
    stopwatch.stop()

    speak_lyrics(lyrics)


if __name__ == '__main__':
    main()

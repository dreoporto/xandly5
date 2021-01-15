
from tensorflow import keras

from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from dre_lib.dre_time import Stopwatch


class LyricsGenerator:

    _PADDING: str = 'pre'

    def __init__(self):
        # TODO AEO load all models/tokenizers on startup
        keras.backend.clear_session()
        self.model = keras.models.load_model('../ai_ml_model/nlp_irish_lit.h5')

    def generate_lyrics(self, seed_text: str, word_group_count: int, words_to_generate: int) -> str:

        catalog = Catalog()
        catalog.add_file_to_catalog('../ai_ml_model/lyrics_files/irish-lyrics-eof.txt')
        max_sequence_length, total_words, features, labels = catalog.tokenize_catalog(self._PADDING)

        lyrics_text = catalog.generate_lyrics_text(
            self.model,
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
    lyrics = generator.generate_lyrics('i wish to see green fields once more',
                                       word_group_count=4, words_to_generate=96)
    print(lyrics)
    stopwatch.stop()


if __name__ == '__main__':
    main()

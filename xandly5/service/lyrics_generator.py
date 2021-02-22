from typing import Dict, List
from tensorflow import keras
from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from xandly5.types.lyrics_model_meta import LyricsModelMeta
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.types.lyrics_section import LyricsSection
from xandly5.types.lyrics_section import LyricsSectionType
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
        self.model_meta: LyricsModelMeta = _lyrics_models[model_id]

    def generate_lyrics(self, seed_text: str, word_group_count: int, word_count: int) -> str:
        lyrics_text = self.model_meta.generate_lyrics_text(seed_text=seed_text, word_count=word_count)
        lyrics_text = LyricsFormatter.format_lyrics(lyrics_text, word_group_count=word_group_count)
        return lyrics_text

    def generate_lyrics_from_sections(self, lyrics_sections: List[LyricsSection]) -> str:
        lyrics_text = ''

        for section in lyrics_sections:
            section.generated_text = self.model_meta.generate_lyrics_text(seed_text=section.seed_text,
                                                                          word_count=section.word_count)
            section.generated_text = LyricsFormatter.format_lyrics(section.generated_text,
                                                                   word_group_count=section.word_group_count)

            # TODO AEO temp debug info
            lyrics_text += f'{section.section_type.name}\n\n' + section.generated_text

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


def create_sonnet():
    stopwatch = Stopwatch()
    lyrics = 'evening fountains lit loss'

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.SONNETS)
    lyrics = generator.generate_lyrics(lyrics, word_group_count=8, word_count=96)
    print(lyrics)
    stopwatch.stop()

    speak_lyrics(lyrics)


def create_poe_poem():
    stopwatch = Stopwatch()
    lyrics = 'deep sleep the oasis'

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.POE_POEM)
    lyrics = generator.generate_lyrics(lyrics, word_group_count=4, word_count=104)
    print(lyrics)
    stopwatch.stop()

    speak_lyrics(lyrics)


def create_structured_lyrics():
    stopwatch = Stopwatch()
    sections: List[LyricsSection] = [
        LyricsSection(section_type=LyricsSectionType.VERSE, word_group_count=4, word_count=32,
                      seed_text='a dreary midnight bird and here i heard'),
        LyricsSection(section_type=LyricsSectionType.CHORUS, word_group_count=4, word_count=16,
                      seed_text='said he art too seas for totter into'),
        LyricsSection(section_type=LyricsSectionType.VERSE, word_group_count=4, word_count=32,
                      seed_text='tone of his eyes of night litten have'),
        LyricsSection(section_type=LyricsSectionType.CHORUS, word_group_count=4, word_count=16,
                      seed_text='said he art too seas for totter into'),
        LyricsSection(section_type=LyricsSectionType.VERSE, word_group_count=4, word_count=32,
                      seed_text='answer step only blows of their harp string'),
        LyricsSection(section_type=LyricsSectionType.BRIDGE, word_group_count=4, word_count=20,
                      seed_text='said he art too'),
        LyricsSection(section_type=LyricsSectionType.OUTRO, word_group_count=4, word_count=16,
                      seed_text='said he art too seas for totter into'),
    ]

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.POE_POEM)
    lyrics = generator.generate_lyrics_from_sections(sections)
    print(lyrics)
    stopwatch.stop()

    # TODO AEO temp replacements
    speak_lyrics(lyrics.replace('VERSE', '').replace('CHORUS', '').replace('BRIDGE', '').replace('OUTRO', ''))


def main():

    # create_sonnet()
    # create_poe_poem()
    create_structured_lyrics()


if __name__ == '__main__':
    main()

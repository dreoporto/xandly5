import os
import re
from typing import Dict, List

from ptmlib.time import Stopwatch
from tensorflow import keras

from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.types.lyrics_model_meta import LyricsModelMeta
from xandly5.types.lyrics_section import LyricsSection
from xandly5.types.section_type_enum import SectionTypeEnum


def _load_lyrics_models() -> Dict[LyricsModelEnum, LyricsModelMeta]:
    print('loading lyrics models')
    models: Dict[LyricsModelEnum, LyricsModelMeta] = {
        LyricsModelEnum.SONNETS: LyricsModelMeta('shakespeare_sonnet.h5', 'shakespeare-sonnets-lyrics.txt'),
        LyricsModelEnum.POE_POEM: LyricsModelMeta('poe_poem.h5', 'poe-poem-lines.txt')
    }

    package_directory = os.path.dirname(os.path.abspath(__file__))

    for _, lyrics_model in models.items():
        lyrics_model.model = keras.models.load_model(
            os.path.join(package_directory, '../ai_ml_model/saved_models/', lyrics_model.model_file))
        lyrics_model.catalog = Catalog()
        lyrics_model.catalog.add_file_to_catalog(
            os.path.join(package_directory, '../ai_ml_model/lyrics_files/', lyrics_model.lyrics_file))
        lyrics_model.catalog.tokenize_catalog()

    return models


# load using global variable
keras.backend.clear_session()
_lyrics_models = _load_lyrics_models()

# validation settings
MAX_SEED_TEXT_LENGTH = 1000
MAX_WORDS_GENERATED = 200


class LyricsGenerator:

    def __init__(self, model_id: LyricsModelEnum):
        print('init LyricsGenerator instance')
        self.model_meta: LyricsModelMeta = _lyrics_models[model_id]

    @staticmethod
    def _clean_seed_text(seed_text: str) -> str:
        seed_text = re.sub('[^A-Za-z0-9 \']+', '', seed_text)  # alphanumeric and ' only
        seed_text = re.sub(' +', ' ', seed_text)  # replace repeating spaces with one space
        seed_text = seed_text.strip()
        return seed_text

    @staticmethod
    def _validate_lyrics_options(seed_text: str, word_group_count: int, word_count: int) -> None:
        if len(seed_text) > MAX_SEED_TEXT_LENGTH:
            raise ValueError(f'Seed Text cannot exceed {MAX_SEED_TEXT_LENGTH} characters')
        if word_count > MAX_WORDS_GENERATED:
            raise ValueError(f'Word Count cannot exceed {MAX_WORDS_GENERATED} words')
        if word_group_count > MAX_WORDS_GENERATED:
            raise ValueError(f'Word Group Count cannot exceed {MAX_WORDS_GENERATED}')

    def generate_lyrics(self, seed_text: str, word_group_count: int, word_count: int) -> str:
        seed_text = self._clean_seed_text(seed_text)
        self._validate_lyrics_options(seed_text, word_group_count, word_count)
        lyrics_text = self.model_meta.generate_lyrics_text(seed_text=seed_text, word_count=word_count)
        lyrics_text = LyricsFormatter.format_lyrics(lyrics_text, word_group_count=word_group_count)
        return lyrics_text

    def generate_lyrics_from_independent_sections(self, lyrics_sections: List[LyricsSection]) -> str:
        lyrics_text = ''

        for section in lyrics_sections:
            section.seed_text = self._clean_seed_text(section.seed_text)
            section.generated_text = self.model_meta.generate_lyrics_text(seed_text=section.seed_text,
                                                                          word_count=section.word_count)
            section.generated_text = LyricsFormatter.format_lyrics(section.generated_text,
                                                                   word_group_count=section.word_group_count)

            lyrics_text += f'--{section.section_type.name}--\n\n' + section.generated_text

        return lyrics_text

    def generate_lyrics_from_sections(self, lyrics_sections: List[LyricsSection]) -> str:
        lyrics_text = ''
        formatted_lyrics_text = ''
        total_word_count = 0

        for section in lyrics_sections:

            total_word_count += section.word_count
            section.seed_text = self._clean_seed_text(section.seed_text)

            # use current lyrics as seed text
            if lyrics_text != '':
                lyrics_text += ' '
            lyrics_text += section.seed_text
            lyrics_text = self.model_meta.generate_lyrics_text(seed_text=lyrics_text,
                                                               word_count=total_word_count)
            # get section text from lyrics, then format
            section.generated_text = self._get_words_at_end(lyrics_text, section.word_count)
            section.generated_text = LyricsFormatter.format_lyrics(section.generated_text,
                                                                   word_group_count=section.word_group_count)

            # will return formatted lyrics with section headers
            formatted_lyrics_text += f'--{section.section_type.name}--\n\n' + section.generated_text

        return formatted_lyrics_text

    @staticmethod
    def _get_words_at_end(text: str, word_count) -> str:
        text_array = text.split(' ')
        end_words_array = text_array[word_count * -1:]
        return ' '.join(end_words_array)


# TODO AEO REMOVE ALL CODE BELOW THIS LINE

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


def create_structured_lyrics():
    stopwatch = Stopwatch()
    sections: List[LyricsSection] = [
        LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                      seed_text='a dreary midnight bird and here i heard'),
        LyricsSection(section_type=SectionTypeEnum.CHORUS, word_group_count=4, word_count=16,
                      seed_text='said he art too seas for totter into'),
        LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                      seed_text='tone of his eyes of night litten have'),
        LyricsSection(section_type=SectionTypeEnum.CHORUS, word_group_count=4, word_count=16,
                      seed_text='said he art too seas for totter into'),
        LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                      seed_text='answer step only blows of their harp string'),
        LyricsSection(section_type=SectionTypeEnum.BRIDGE, word_group_count=4, word_count=20,
                      seed_text='said he art too'),
        LyricsSection(section_type=SectionTypeEnum.OUTRO, word_group_count=4, word_count=16,
                      seed_text='said he art too seas for totter into'),
    ]

    stopwatch.start()
    generator = LyricsGenerator(LyricsModelEnum.POE_POEM)
    lyrics = generator.generate_lyrics_from_sections(sections)
    print(lyrics)
    stopwatch.stop()

    lyrics = re.sub('--[A-Z]+--', '', lyrics)
    speak_lyrics(lyrics)


def main():
    create_structured_lyrics()


if __name__ == '__main__':
    main()

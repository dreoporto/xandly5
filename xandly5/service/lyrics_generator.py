import json
import os
import re
from typing import Dict, List

from tensorflow import keras

from xandly5.ai_ml_model.catalog import Catalog
from xandly5.ai_ml_model.lyrics_formatter import LyricsFormatter
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.types.lyrics_model_meta import LyricsModelMeta
from xandly5.types.lyrics_section import LyricsSection
from xandly5.types.validation_error import ValidationError


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


class LyricsGenerator:

    _package_directory = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(_package_directory, 'lyrics_generator_config.json')) as json_file:
        _config = json.load(json_file)

    max_seed_text_length = int(_config['max_seed_text_length'])
    max_words_generated = int(_config['max_words_generated'])
    max_lyrics_sections = int(_config['max_lyrics_sections'])

    def __init__(self, model_id: LyricsModelEnum):
        """
        Provices logic for generating lyrics using pre-trained models, including input validations

        :param model_id: model to use for word generation (LyricsModelEnum)
        """

        if model_id not in _lyrics_models:
            raise ValidationError(f'Invalid Model Id: {model_id}')
        print('init LyricsGenerator instance')
        self.model_meta: LyricsModelMeta = _lyrics_models[model_id]

    @staticmethod
    def _clean_seed_text(seed_text: str) -> str:
        seed_text = re.sub('[^A-Za-z0-9 \']+', '', seed_text)  # alphanumeric and ' only
        seed_text = re.sub(' +', ' ', seed_text)  # replace repeating spaces with one space
        seed_text = seed_text.strip()
        return seed_text

    def _validate_lyrics_options(self, seed_text: str, word_group_count: int, word_count: int) -> None:
        if not isinstance(word_count, int):
            raise ValidationError(f'Word Count value is invalid')
        if not isinstance(word_group_count, int):
            raise ValidationError(f'Word Group Count value is invalid')
        if len(seed_text) > self.max_seed_text_length:
            raise ValidationError(f'Seed Text cannot exceed {self.max_seed_text_length} characters')
        if word_count > self.max_words_generated:
            raise ValidationError(f'Word Count cannot exceed {self.max_words_generated}')
        if word_group_count > self.max_words_generated:
            raise ValidationError(f'Word Group Count cannot exceed {self.max_words_generated}')

    def _clean_and_validate_lyrics_sections(self, lyrics_sections: List[LyricsSection]) -> None:
        if len(lyrics_sections) > self.max_lyrics_sections:
            raise ValidationError(f'Total number of Lyrics Sections cannot exceed {self.max_lyrics_sections}')
        for section in lyrics_sections:
            section.seed_text = self._clean_seed_text(section.seed_text)
            self._validate_lyrics_options(seed_text=section.seed_text, word_group_count=section.word_group_count,
                                          word_count=section.word_count)

    def generate_lyrics(self, seed_text: str, word_group_count: int, word_count: int) -> str:
        """
        creates lyrics using the specified starter text

        :param seed_text: starter text
        :param word_group_count: controls the addition of commas or blank lines
        :param word_count: total number of words to return
        :return: seed text + generated text
        """
        seed_text = self._clean_seed_text(seed_text)
        self._validate_lyrics_options(seed_text=seed_text, word_group_count=word_group_count, word_count=word_count)
        lyrics_text = self.model_meta.generate_lyrics_text(seed_text=seed_text, word_count=word_count)
        lyrics_text = LyricsFormatter.format_lyrics(lyrics_text, word_group_count=word_group_count)
        return lyrics_text

    def generate_lyrics_from_independent_sections(self, lyrics_sections: List[LyricsSection]) -> str:
        """
        creates lyrics using a LyricsSection list; sections are *not* influenced by the text in other sections

        :param lyrics_sections: list of LyricsSection
        :return: all lyrics, formatted
        """

        lyrics_text = ''

        self._clean_and_validate_lyrics_sections(lyrics_sections)

        for section in lyrics_sections:
            section.generated_text = self.model_meta.generate_lyrics_text(seed_text=section.seed_text,
                                                                          word_count=section.word_count)
            section.generated_text = LyricsFormatter.format_lyrics(section.generated_text,
                                                                   word_group_count=section.word_group_count)

            lyrics_text += f'--{section.section_type.name}--\n\n' + section.generated_text

        return lyrics_text

    def generate_lyrics_from_sections(self, lyrics_sections: List[LyricsSection]) -> str:
        """
        creates lyrics using a LyricsSection list; later sections are influenced by the text in previous sections

        :param lyrics_sections:
        :return: all lyrics, formatted
        """

        lyrics_text = ''
        formatted_lyrics_text = ''
        total_word_count = 0

        self._clean_and_validate_lyrics_sections(lyrics_sections)

        for section in lyrics_sections:
            total_word_count += section.word_count

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

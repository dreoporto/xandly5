import hashlib
import os
import re
import unittest
from typing import List

from ptmlib.time import Stopwatch

from xandly5.service.lyrics_generator import LyricsGenerator
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.types.lyrics_section import LyricsSection
from xandly5.types.section_type_enum import SectionTypeEnum
from xandly5.types.validation_error import ValidationError


class LyricsGeneratorStructuredTestCase(unittest.TestCase):

    # ENABLE THIS WHEN CREATING NEW TESTS
    WRITE_LYRICS_FILES: bool = False

    @staticmethod
    def _get_hash(value: str) -> str:
        return hashlib.md5(value.encode('utf-8')).hexdigest()

    @staticmethod
    def _get_lyrics_sections() -> List[LyricsSection]:
        sections: List[LyricsSection] = [
            LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                          seed_text='a dreary  midnight   bir!d and here </> i heard'),
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

        return sections

    def test_generate_structured_lyrics(self):

        # ARRANGE
        expected_lyrics_file = 'expected_poe_struct_lyrics.txt'
        model_id = LyricsModelEnum.POE_POEM

        lyrics_sections = self._get_lyrics_sections()

        # ACT, ASSERT
        self.generate_lyrics_for_test(expected_lyrics_file, lyrics_sections, model_id)

    def test_generate_structured_lyrics_independent_sections(self):

        # ARRANGE
        expected_lyrics_file = 'expected_sonnet_struct_lyrics.txt'
        model_id = LyricsModelEnum.SONNETS

        lyrics_sections = self._get_lyrics_sections()

        # ACT, ASSERT
        self.generate_lyrics_for_test(expected_lyrics_file, lyrics_sections, model_id, independent_sections=True)

    def test_generate_structured_lyrics_error(self):

        lyrics_sections: List[LyricsSection] = []
        for _ in range(0, LyricsGenerator.max_lyrics_sections + 1):
            lyrics_sections.append(
                LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=20,
                              seed_text='hello')
            )

        self.assertRaises(ValidationError, self.generate_lyrics_for_test, 'expected_sonnet_struct_lyrics.txt',
                          lyrics_sections, LyricsModelEnum.SONNETS, True)

        self.assertRaises(ValidationError, self.generate_lyrics_for_test, 'expected_sonnet_struct_lyrics.txt',
                          lyrics_sections, LyricsModelEnum.SONNETS, False)

        short_list = lyrics_sections[0:1]
        short_list[0].word_count = LyricsGenerator.max_words_generated + 1

        self.assertRaises(ValidationError, self.generate_lyrics_for_test, 'expected_sonnet_struct_lyrics.txt',
                          short_list, LyricsModelEnum.SONNETS, False)

    def generate_lyrics_for_test(self, expected_lyrics_file: str, lyrics_sections: List[LyricsSection],
                                 model_id: LyricsModelEnum, independent_sections: bool = False) -> None:

        # ACT
        stopwatch = Stopwatch()
        stopwatch.start()
        generator = LyricsGenerator(model_id)

        lyrics: str
        if independent_sections:
            lyrics = generator.generate_lyrics_from_independent_sections(lyrics_sections)
        else:
            lyrics = generator.generate_lyrics_from_sections(lyrics_sections)

        stopwatch.stop()

        current_directory = os.path.dirname(os.path.abspath(__file__))

        # ASSERT
        if self.WRITE_LYRICS_FILES:
            with open(os.path.join(current_directory, expected_lyrics_file), 'w') as file:
                file.write(lyrics)
        with open(os.path.join(current_directory, expected_lyrics_file), 'r') as file:
            expected_lyrics = file.read()

        clean_lyrics = re.sub('--[A-Z]+--', '', lyrics)  # remove section headers
        clean_lyrics = clean_lyrics.strip().replace('  ', ' ').replace(',', '')  # removes unnecessary chars
        total_word_count: int = sum([s.word_count for s in lyrics_sections])

        self.assertEqual(total_word_count, len(clean_lyrics.split(' ')))
        for section in lyrics_sections:
            clean_section_generated = section.generated_text.replace('\n', '').replace(',', '').replace('  ', ' ')
            self.assertTrue(clean_section_generated.startswith(section.seed_text))
        self.assertEqual(self._get_hash(expected_lyrics), self._get_hash(lyrics))


if __name__ == '__main__':
    unittest.main()

import unittest
from typing import List
import hashlib

from ptmlib.time import Stopwatch

from xandly5.types.lyrics_section import LyricsSection
from xandly5.types.section_type_enum import SectionTypeEnum
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.service.lyrics_generator import LyricsGenerator


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

        return sections

    def test_generate_structured_lyrics(self):

        # ARRANGE
        expected_lyrics_file = 'expected_poe_struct_lyrics.txt'
        model_id = LyricsModelEnum.POE_POEM

        lyrics_sections = self._get_lyrics_sections()
        total_word_count: int = sum([s.word_count for s in lyrics_sections])

        # ACT
        stopwatch = Stopwatch()
        stopwatch.start()
        generator = LyricsGenerator(model_id)
        lyrics = generator.generate_lyrics_from_sections(lyrics_sections)
        stopwatch.stop()

        # ASSERT
        if self.WRITE_LYRICS_FILES:
            with open(expected_lyrics_file, 'w') as file:
                file.write(lyrics)
        with open(expected_lyrics_file, 'r') as file:
            expected_lyrics = file.read()

        clean_lyrics = lyrics.strip().replace('  ', ' ').replace(',', '')  # removes unnecessary chars

        self.assertEqual(total_word_count, len(clean_lyrics.split(' ')))
        for section in lyrics_sections:
            clean_section_generated = section.generated_text.replace('\n', '').replace(',', '').replace('  ', ' ')
            self.assertTrue(clean_section_generated.startswith(section.seed_text))
        self.assertEqual(self._get_hash(expected_lyrics), self._get_hash(lyrics))


if __name__ == '__main__':
    unittest.main()

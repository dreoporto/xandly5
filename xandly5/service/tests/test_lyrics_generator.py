import unittest
from typing import List
import hashlib

from ptmlib.time import Stopwatch

from xandly5.types.lyrics_section import LyricsSection
from xandly5.types.section_type_enum import SectionTypeEnum
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.service.lyrics_generator import LyricsGenerator


class LyricsGeneratorTestCase(unittest.TestCase):

    @staticmethod
    def _get_hash(value: str) -> str:
        return hashlib.md5(value.encode('utf-8')).hexdigest()

    @staticmethod
    def _get_lyrics_sections():
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

    def test_generate_sonnet_lyrics(self):

        # ARRANGE
        stopwatch = Stopwatch()
        word_count = 96
        seed_text = 'evening fountains lit loss'

        # ACT
        stopwatch.start()
        generator = LyricsGenerator(LyricsModelEnum.SONNETS)
        lyrics = generator.generate_lyrics(seed_text=seed_text, word_group_count=8, word_count=word_count)
        stopwatch.stop()

        # ASSERT
        clean_lyrics = lyrics.strip().replace('  ', ' ').replace(',', '')  # removes unnecessary chars

        with open('expected_sonnet_lyrics.txt', 'r') as file:
            expected_lyrics = file.read()

        self.assertEqual(word_count, len(clean_lyrics.split(' ')))
        self.assertTrue(lyrics.startswith(seed_text))
        self.assertEqual(self._get_hash(expected_lyrics), self._get_hash(lyrics))


if __name__ == '__main__':
    unittest.main()

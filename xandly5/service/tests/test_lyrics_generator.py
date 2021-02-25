import unittest
import hashlib

from ptmlib.time import Stopwatch

from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.service.lyrics_generator import LyricsGenerator


class LyricsGeneratorTestCase(unittest.TestCase):

    # ENABLE THIS WHEN CREATING NEW TESTS
    WRITE_LYRICS_FILES: bool = False

    @staticmethod
    def _get_hash(value: str) -> str:
        return hashlib.md5(value.encode('utf-8')).hexdigest()

    def test_generate_sonnet_lyrics(self):

        # ARRANGE
        word_count = 96
        word_group_count = 8
        seed_text = 'evening fountains lit loss'
        expected_lyrics_file = 'expected_sonnet_lyrics.txt'
        model_id = LyricsModelEnum.SONNETS

        # ACT, ASSERT
        self.generate_lyrics_for_test(expected_lyrics_file, model_id, seed_text, word_count, word_group_count)

    def test_generate_poe_lyrics(self):

        # ARRANGE
        word_count = 100
        word_group_count = 4
        seed_text = 'a dreary midnight bird'
        expected_lyrics_file = 'expected_poe_lyrics.txt'
        model_id = LyricsModelEnum.POE_POEM

        # ACT, ASSERT
        self.generate_lyrics_for_test(expected_lyrics_file, model_id, seed_text, word_count, word_group_count)

    def generate_lyrics_for_test(self, expected_lyrics_file, model_id, seed_text, word_count, word_group_count):

        # ACT
        stopwatch = Stopwatch()
        stopwatch.start()
        generator = LyricsGenerator(model_id)
        lyrics = generator.generate_lyrics(seed_text=seed_text, word_group_count=word_group_count,
                                           word_count=word_count)
        stopwatch.stop()

        # ASSERT
        clean_lyrics = lyrics.strip().replace('  ', ' ').replace(',', '')  # removes unnecessary chars

        if self.WRITE_LYRICS_FILES:
            with open(expected_lyrics_file, 'w') as file:
                file.write(lyrics)
        with open(expected_lyrics_file, 'r') as file:
            expected_lyrics = file.read()
        self.assertEqual(word_count, len(clean_lyrics.split(' ')))
        self.assertTrue(lyrics.startswith(seed_text))
        self.assertEqual(self._get_hash(expected_lyrics), self._get_hash(lyrics))


if __name__ == '__main__':
    unittest.main()
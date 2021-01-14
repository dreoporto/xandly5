
class LyricsFormatter:

    @staticmethod
    def format_lyrics(lyric_text: str, word_group_count: int = 4):
        """
        format lyrics into a more human-readable layout

        :param lyric_text: text to be formatted
        :param word_group_count: number of words to group by for formatting

        :return: formatted lyric text
        """
        words = lyric_text.split(' ')
        formatted_lyrics = ''

        word_index = 0

        for word in words:
            word_index += 1
            formatted_lyrics += word
            if word_index % (word_group_count * 2) == 0:
                formatted_lyrics += ' \n\n'
            elif word_index % word_group_count == 0:
                formatted_lyrics += ',\n  '
            else:
                formatted_lyrics += ' '

        return formatted_lyrics

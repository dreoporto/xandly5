
"""
CONVERTS RAW TEXT OF SHAKESPEARE'S SONNETS TO A TAB-SEPARATED DATA FILE
ALSO GENERATES A FILE WITH LYRICS ONLY, FOR NLP USAGE

TEXT	SONNET NUMBER	LINE
From fairest creatures we desire increase,	1	1
That thereby beauty's rose might never die,	1	2
"""

import re
import string


def convert_text_to_data():

    sonnet_number: int = 0
    line_number: int = 0
    # pattern for sonnet number as roman numeral; only goes up to 154 so no need for M as in MCMLXXXIV
    roman_num_regexp = re.compile('^[CLXVI]*$')

    with open('shakespeare-sonnets.txt', 'r') as sonnets_file:
        with open('shakespeare-sonnets-data.txt', 'w') as sonnets_data:

            sonnets_data.write('TEXT\tSONNET NUMBER\tLINE\n')

            for line in sonnets_file:

                # strip whitespace
                clean_line = line.strip()
                # strip punctuation at BEGINNING and END of words; does not change words such as: beauty's and o'er
                clean_line = ' '.join([word.strip(string.punctuation) for word in clean_line.split(" ")])

                if len(clean_line) == 0:
                    # skip blank line
                    continue
                elif roman_num_regexp.search(clean_line):
                    # sonnet number indicator; increment sonnet number and RESET line count
                    sonnet_number += 1
                    line_number = 0
                else:
                    line_number += 1
                    sonnets_data.write(f'{clean_line}\t{sonnet_number}\t{line_number}\n')


def convert_data_to_lyrics():

    with open('shakespeare-sonnets-data.txt', 'r') as sonnets_data:
        with open('shakespeare-sonnets-lyrics.txt', 'w') as sonnets_lyrics:
            next(sonnets_data)  # skip first line
            for line in sonnets_data:
                lyrics = line.split('\t')[0]
                sonnets_lyrics.write(f'{lyrics}\n')


def main():
    convert_text_to_data()
    convert_data_to_lyrics()


if __name__ == '__main__':
    main()

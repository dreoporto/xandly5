
"""
CONVERTS RAW TEXT OF SHAKESPEARE'S SONNETS TO A TAB-SEPARATED DATA FILE

TEXT	SONNET NUMBER	LINE
From fairest creatures we desire increase,	1	1
That thereby beauty's rose might never die,	1	2
"""

import re


def main():

    sonnet_number: int = 0
    line_number: int = 0
    # pattern for sonnet number as roman numeral; only goes up to 154 so no need for M as in MCMLXXXIV
    roman_num_regexp = re.compile('^[CLXVI]*$')

    with open('shakespeare-sonnets.txt', 'r') as sonnets_file:
        with open('shakespeare-sonnets-data.txt', 'w') as sonnets_data:

            sonnets_data.write('TEXT\tSONNET NUMBER\tLINE\n')

            for line in sonnets_file:

                clean_line = line.strip()

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


if __name__ == '__main__':
    main()

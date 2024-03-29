import numpy as np
import csv
from typing import List, Optional

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Catalog:
    """
    represents a catalog (aka corpus) of works, used in both model training and prediction
    """

    def __init__(self, padding: str = 'pre', oov_token='<OOV>'):
        self.catalog_items: List[str] = []
        self.tokenizer = Tokenizer(oov_token=oov_token)
        self.max_sequence_length = 0
        self.total_words = 0
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._padding = padding

    def add_file_to_catalog(self, file_name: str) -> None:
        """
        add a text file to the catalog

        :param file_name: file name with lyrics/text
        :return: None
        """
        with open(file_name, 'r') as text_file:
            for line in text_file:
                self.catalog_items.append(line.lower())

    def add_csv_file_to_catalog(self, file_name: str, text_column: int, skip_first_line: bool = True,
                                delimiter: str = ',') -> None:
        """
        add a csv, tsv or other delimited file to the catalog

        :param file_name: file name with lyrics/text
        :param text_column: column number to select, 0 based
        :param skip_first_line: skip first line of text
        :param delimiter: delimiter to use as separator
        :return: None
        """
        with open(file_name, 'r') as text_file:
            csv_reader = csv.reader(text_file, delimiter=delimiter)
            if skip_first_line:
                next(csv_reader)
            for row in csv_reader:
                self.catalog_items.append(row[text_column])

    def tokenize_catalog(self) -> None:

        """
        tokenize the contents of the catalog, and set properties accordingly (ex: total_words, labels)

        :return: None
        """

        # tokenizer: fit, sequence, pad
        self.tokenizer.fit_on_texts(self.catalog_items)

        # create a list of n-gram sequences
        input_sequences = []

        for line in self.catalog_items:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        # pad sequences
        self.max_sequence_length = max([len(item) for item in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_length,
                                                 padding=self._padding))

        self.features = input_sequences[:, :-1]
        labels_temp = input_sequences[:, -1]

        self.total_words = len(self.tokenizer.word_index) + 1
        self.labels = keras.utils.to_categorical(labels_temp, num_classes=self.total_words)

        # for troubleshooting only! output word index
        # print(sorted(self.tokenizer.word_index.keys()))

    def generate_lyrics_text(self, model: keras.Sequential, seed_text: str, word_count: int) -> str:

        """
        generate lyrics using the provided model and properties

        :param model: model used to generate text
        :param seed_text: starter text
        :param word_count: total number of words to return
        :return: starter text + generated text
        """

        seed_text_word_count = len(seed_text.split(' '))
        words_to_generate = word_count - seed_text_word_count

        for _ in range(words_to_generate):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length - 1, padding=self._padding)
            predicted = np.argmax(model.predict(token_list), axis=-1)

            output_word = self.tokenizer.index_word[int(predicted)]
            if output_word is not None:
                seed_text += ' ' + output_word

        return seed_text

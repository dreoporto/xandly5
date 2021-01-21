
import numpy as np
from typing import List, Optional

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Catalog:

    """
    represents a catalog (aka corpus) of works
    """

    def __init__(self, padding: str = 'pre'):
        self.catalog_items: List[str] = []
        self.tokenizer = Tokenizer()
        self.max_sequence_length = 0
        self.total_words = 0
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._padding = padding

    def add_file_to_catalog(self, file_name: str):
        with open(file_name) as text_file:
            for line in text_file:
                self.catalog_items.append(line.lower())

    def tokenize_catalog(self) -> None:

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

    def generate_lyrics_text(self, model: keras.Sequential, seed_text: str,
                             word_count: int, max_sequence_length: int) -> str:

        for _ in range(word_count):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding=self._padding)
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ''

            # TODO AEO there's a faster way to do this
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break

            seed_text += ' ' + output_word

        return seed_text

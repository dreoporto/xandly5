import json
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split

from ptmlib.time import Stopwatch
from ptmlib import charts as pch
from catalog import Catalog
from lyrics_formatter import LyricsFormatter


def tensorflow_diagnostics():
    print('tf version:', tf.__version__)
    print('keras version:', keras.__version__)


class PoePoemModel:

    def __init__(self, config_file):
        with open(config_file) as json_file:
            self.config = json.load(json_file)

    @staticmethod
    def _get_model(total_words: int, max_sequence_length: int,
                   output_dimensions: int, lstm_units: int) -> keras.Sequential:

        model = keras.Sequential([
            keras.layers.Embedding(total_words, output_dimensions, input_length=max_sequence_length - 1),
            keras.layers.Bidirectional(keras.layers.LSTM(lstm_units)),
            keras.layers.Dense(total_words, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        return model

    def _generate_lyrics(self, model: keras.Sequential, catalog: Catalog):

        lyrics_text = catalog.generate_lyrics_text(
            model,
            seed_text=self.config['seed_text'],
            word_count=self.config['words_to_generate']
        )

        lyrics = LyricsFormatter.format_lyrics(lyrics_text, self.config['word_group_count'])
        print(lyrics)

        with open(self.config['saved_lyrics_path'], 'w') as lyrics_file:
            lyrics_file.write(lyrics)
            lyrics_file.close()

    def train_model(self):

        catalog = Catalog()
        catalog.add_file_to_catalog(self.config['lyrics_file_path'])
        catalog.tokenize_catalog()

        x_train, x_valid, y_train, y_valid = train_test_split(
            catalog.features, catalog.labels,
            test_size=self.config['hp_test_size'],
            random_state=self.config['random_state']
        )

        model = self._get_model(
            max_sequence_length=catalog.max_sequence_length,
            total_words=catalog.total_words,
            output_dimensions=self.config['hp_output_dimensions'],
            lstm_units=self.config['hp_lstm_units']
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['hp_patience'],
            min_delta=self.config['hp_min_delta'],
            mode='min'
        )

        stopwatch = Stopwatch()
        stopwatch.start()

        history = model.fit(
            catalog.features,
            catalog.labels,
            validation_data=(x_valid, y_valid),
            epochs=self.config['hp_epochs'],
            verbose=1,
            callbacks=[early_stopping]
        )

        stopwatch.stop()

        model.save(self.config['saved_model_path'])

        if self.config['is_interactive']:
            pch.show_history_chart(history, 'accuracy')
            pch.show_history_chart(history, 'loss')

        self._generate_lyrics(model, catalog)


def main():
    tensorflow_diagnostics()
    lyrics_model = PoePoemModel('poe_poem_config.json')
    lyrics_model.train_model()


if __name__ == '__main__':
    main()

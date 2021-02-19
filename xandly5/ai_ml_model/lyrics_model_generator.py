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


class LyricsModelGenerator:

    def __init__(self, config_file):
        with open(config_file) as json_file:
            self.config = json.load(json_file)
        self.catalog = Catalog()
        self.catalog.add_file_to_catalog(self.config['lyrics_file_path'])
        self.catalog.tokenize_catalog()
        self.is_interactive = self.config['is_interactive']

    def _get_compiled_model(self) -> keras.Sequential:

        total_words = self.catalog.total_words
        dimensions = self.config['hp_output_dimensions']
        input_length = self.catalog.max_sequence_length - 1
        units = self.config['hp_lstm_units']

        model = keras.Sequential([
            keras.layers.Embedding(total_words, dimensions, input_length=input_length),
            keras.layers.Bidirectional(keras.layers.LSTM(units)),
            keras.layers.Dense(total_words, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if self.is_interactive:
            model.summary()

        return model

    def _train_model(self, model: keras.Sequential):

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['hp_patience'],
            min_delta=self.config['hp_min_delta'],
            mode='min'
        )

        stopwatch = Stopwatch()
        stopwatch.start()

        x_train, x_valid, y_train, y_valid = train_test_split(
            self.catalog.features, self.catalog.labels,
            test_size=self.config['hp_test_size'],
            random_state=self.config['random_state']
        )

        history = model.fit(
            self.catalog.features,
            self.catalog.labels,
            validation_data=(x_valid, y_valid),
            epochs=self.config['hp_epochs'],
            verbose=1,
            callbacks=[early_stopping]
        )

        stopwatch.stop(silent=not self.is_interactive)

        if self.is_interactive:
            pch.show_history_chart(history, 'accuracy')
            pch.show_history_chart(history, 'loss')

    def _generate_sample_lyrics(self, model: keras.Sequential):

        lyrics_text = self.catalog.generate_lyrics_text(
            model,
            seed_text=self.config['seed_text'],
            word_count=self.config['words_to_generate']
        )

        lyrics = LyricsFormatter.format_lyrics(lyrics_text, self.config['word_group_count'])
        print(lyrics)

        with open(self.config['saved_lyrics_path'], 'w') as lyrics_file:
            lyrics_file.write(lyrics)
            lyrics_file.close()

    def generate_model(self):

        model = self._get_compiled_model()
        self._train_model(model)
        model.save(self.config['saved_model_path'])
        self._generate_sample_lyrics(model)

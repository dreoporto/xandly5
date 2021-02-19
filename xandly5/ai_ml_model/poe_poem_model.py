import os
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


def get_model(total_words: int, max_sequence_length: int,
              output_dimensions: int, lstm_units: int) -> keras.Sequential:
    model = keras.Sequential([
        keras.layers.Embedding(total_words, output_dimensions, input_length=max_sequence_length - 1),
        keras.layers.Bidirectional(keras.layers.LSTM(lstm_units)),
        keras.layers.Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def main():
    tensorflow_diagnostics()

    with open('poe_poem_config.json') as json_file:
        config = json.load(json_file)

    catalog = Catalog()
    catalog.add_file_to_catalog(config['lyrics_file_path'])
    catalog.tokenize_catalog()

    x_train, x_valid, y_train, y_valid = train_test_split(
        catalog.features, catalog.labels,
        test_size=config['hp_test_size'],
        random_state=config['random_state']
    )

    model = get_model(
        max_sequence_length=catalog.max_sequence_length,
        total_words=catalog.total_words,
        output_dimensions=config['hp_output_dimensions'],
        lstm_units=config['hp_lstm_units']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['hp_patience'],
        min_delta=config['hp_min_delta'],
        mode='min'
    )

    stopwatch = Stopwatch()
    stopwatch.start()

    history = model.fit(
        catalog.features,
        catalog.labels,
        validation_data=(x_valid, y_valid),
        epochs=config['hp_epochs'],
        verbose=1,
        callbacks=[early_stopping]
    )

    stopwatch.stop()

    model.save(config['saved_model_path'])

    pch.show_history_chart(history, 'accuracy')
    pch.show_history_chart(history, 'loss')

    lyrics_text = catalog.generate_lyrics_text(
        model,
        seed_text=config['seed_text'],
        word_count=config['words_to_generate']
    )

    lyrics = LyricsFormatter.format_lyrics(lyrics_text, config['word_group_count'])
    print(lyrics)

    with open(config['saved_lyrics_path'], 'w') as lyrics_file:
        lyrics_file.write(lyrics)
        lyrics_file.close()


if __name__ == '__main__':
    main()

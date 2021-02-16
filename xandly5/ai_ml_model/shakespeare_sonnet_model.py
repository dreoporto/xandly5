import os
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

    # HYPER PARAMS
    hp_epochs = 150
    hp_output_dimensions = 100
    hp_lstm_units = 150
    hp_patience = 8
    hp_min_delta = 0.001

    # other params
    lyrics_dir = 'lyrics_files'
    word_group_count = 4
    seed_text = 'andre went to stay on the frozen shore'
    words_to_generate = 100
    random_state = 42

    catalog = Catalog()
    catalog.add_csv_file_to_catalog(os.path.join(lyrics_dir, 'shakespeare-sonnets-data.txt'), text_column=0,
                                    delimiter='\t')
    catalog.tokenize_catalog()

    x_train, x_valid, y_train, y_valid = train_test_split(catalog.features, catalog.labels,
                                                          test_size=0.3, random_state=random_state)

    model = get_model(
        max_sequence_length=catalog.max_sequence_length,
        total_words=catalog.total_words,
        output_dimensions=hp_output_dimensions,
        lstm_units=hp_lstm_units
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=hp_patience,
        min_delta=hp_min_delta,
        mode='min'
    )

    stopwatch = Stopwatch()
    stopwatch.start()

    history = model.fit(
        catalog.features,
        catalog.labels,
        validation_data=(x_valid, y_valid),
        epochs=hp_epochs,
        verbose=1,
        callbacks=[early_stopping]
    )

    stopwatch.stop()

    model.save('saved_models/shakespeare_sonnet.h5')

    pch.show_history_chart(history, 'accuracy')
    pch.show_history_chart(history, 'loss')

    lyrics_text = catalog.generate_lyrics_text(
        model,
        seed_text=seed_text,
        word_count=words_to_generate
    )

    lyrics = LyricsFormatter.format_lyrics(lyrics_text, word_group_count)
    print(lyrics)

    with open('saved_models/shakespeare_sonnet_new_lyrics.txt', 'w') as lyrics_file:
        lyrics_file.write(lyrics)
        lyrics_file.close()


if __name__ == '__main__':
    main()

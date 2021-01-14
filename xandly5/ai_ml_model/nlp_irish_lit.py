import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from dre_lib.dre_time import Stopwatch
from dre_lib import dre_chartz as dc
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


# TODO AEO MOVE THIS OUT
def speak_lyrics(lyrics: str):
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    engine.setProperty('rate', 160)
    # noinspection PyUnresolvedReferences
    engine.setProperty('voice', voices[1].id)

    engine.say(lyrics)

    engine.runAndWait()
    engine.stop()


def main():

    tensorflow_diagnostics()

    # HYPER PARAMS
    hp_padding = 'pre'
    hp_epochs = 150
    hp_output_dimensions = 100
    hp_lstm_units = 150
    hp_patience = 8
    hp_min_delta = 0.001

    # other params
    lyrics_dir = 'lyrics_files'
    word_group_count = 4
    seed_text = 'andre went to dublin looking for a breakdown'
    words_to_generate = 100
    random_state = 42

    catalog = Catalog()
    catalog.add_file_to_catalog(os.path.join(lyrics_dir, 'irish-lyrics-eof.txt'))
    max_sequence_length, total_words, features, labels = catalog.tokenize_catalog(hp_padding)

    x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.3, random_state=random_state)

    model = get_model(
        max_sequence_length=max_sequence_length,
        total_words=total_words,
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
        features,
        labels,
        validation_data=(x_valid, y_valid),
        epochs=hp_epochs,
        verbose=1,
        callbacks=[early_stopping]
    )

    stopwatch.stop()

    model.save('nlp_irish_lit.h5')

    dc.show_history_chart(history, 'accuracy')
    dc.show_history_chart(history, 'loss')

    lyrics_text = catalog.generate_lyrics_text(
        model,
        seed_text=seed_text,
        padding=hp_padding,
        word_count=words_to_generate,
        max_sequence_length=max_sequence_length
    )

    lyrics = LyricsFormatter.format_lyrics(lyrics_text, word_group_count)
    print(lyrics)

    with open('nlp_irish_lit_new_lyrics.txt', 'w') as lyrics_file:
        lyrics_file.write(lyrics)
        lyrics_file.close()

    # SPEAK, POET, SPEAK!
    # speak_lyrics(lyrics)  # TODO AEO TEMP


if __name__ == '__main__':
    main()

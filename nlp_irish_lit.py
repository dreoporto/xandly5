import os
import numpy as np
from typing import List


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dre_lib.dre_time import Stopwatch
from dre_lib import dre_chartz as dc


def tensorflow_diagnostics():
    print('tf version:', tf.__version__)
    print('keras version:', keras.__version__)


def add_file_to_corpus(file_name: str, corpus: List[str]):
    with open(file_name) as text_file:
        for line in text_file:
            corpus.append(line.lower())


def format_lyrics(lyric_text: str, word_group_count: int = 4):
    """
    format lyrics into a more human-readable layout

    :param lyric_text:
    :param word_group_count:
    :return:
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

    # HYPER PARAMS
    hp_padding = 'pre'
    hp_epochs = 250

    # OTHER PARAMS
    lyrics_dir = 'lyrics_files'
    word_group_count = 4
    seed_text = 'andre went to dublin looking for a breakdown'
    generate_words_quantity = 100

    stopwatch = Stopwatch()

    tensorflow_diagnostics()

    # GET TEXT DATA

    corpus = []
    add_file_to_corpus(os.path.join(lyrics_dir, 'communication-breakdown.txt'), corpus)
    add_file_to_corpus(os.path.join(lyrics_dir, 'lanigans-ball.txt'), corpus)

    # TOKENIZER: FIT, SEQUENCE, PAD

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1
    print(list(tokenizer.word_index.items())[:10])
    print('\n', total_words)

    # CREATE A LIST OF N-GRAM SEQUENCES

    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # PAD SEQUENCES

    max_sequence_length = max([len(item) for item in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding=hp_padding))

    # SETUP FEATURES AND LABELS

    xs = input_sequences[:, :-1]
    labels = input_sequences[:, -1]
    ys = keras.utils.to_categorical(labels, num_classes=total_words)

    # TODO AEO TEMP - CHECK DATA
    # print(xs[6], '\n')
    # print(ys[6], '\n')
    # print(np.where(ys[6] == 1)[0][0])

    # CREATE MODEL

    model = keras.Sequential([
        keras.layers.Embedding(total_words, 64, input_length=max_sequence_length - 1),
        keras.layers.Bidirectional(keras.layers.LSTM(20)),
        keras.layers.Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    stopwatch.start()

    history = model.fit(
        xs,
        ys,
        epochs=hp_epochs,
        verbose=1
    )

    stopwatch.stop()

    dc.show_history_chart(history, 'accuracy')
    dc.show_history_chart(history, 'loss')

    # GENERATE TEXT

    for _ in range(generate_words_quantity):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding=hp_padding)
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ''

        # TODO AEO there's a faster way to do this
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += ' ' + output_word

    lyrics = format_lyrics(seed_text, word_group_count)

    print(lyrics)

    # SPEAK, POET, SPEAK!
    # speak_lyrics(lyrics)  # TODO AEO TEMP


if __name__ == '__main__':
    main()

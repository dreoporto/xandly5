
import lyrics_model as lm
from tensorflow import keras


class ShakespeareSonnetModel(lm.LyricsModel):

    def __init__(self, config_file: str):
        super().__init__(config_file)

    def _get_compiled_model(self) -> keras.Sequential:

        model = keras.Sequential([
            keras.layers.Embedding(self.catalog.total_words, self.config['hp_output_dimensions'],
                                   input_length=self.catalog.max_sequence_length - 1),
            keras.layers.Bidirectional(keras.layers.LSTM(self.config['hp_lstm_units'], return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(self.config['hp_lstm_units'])),
            keras.layers.Dense(self.catalog.total_words, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if self.is_interactive:
            model.summary()

        return model


def main():
    lm.tensorflow_diagnostics()
    lyrics_model = ShakespeareSonnetModel('shakespeare_sonnet_config.json')
    lyrics_model.generate_model()


if __name__ == '__main__':
    main()

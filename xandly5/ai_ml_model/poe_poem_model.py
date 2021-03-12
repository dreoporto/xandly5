
import lyrics_model as lm


def main():
    lm.tensorflow_diagnostics()
    lyrics_model = lm.LyricsModel('poe_poem_config.json')
    lyrics_model.generate_model()


if __name__ == '__main__':
    main()

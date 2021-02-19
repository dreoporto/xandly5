
import lyrics_model_generator as lmg


def main():
    lmg.tensorflow_diagnostics()
    lyrics_model = lmg.LyricsModelGenerator('poe_poem_config.json')
    lyrics_model.generate_model()


if __name__ == '__main__':
    main()


from flask import Flask, request, make_response
from flask_restful import Resource, Api

from xandly5.service.lyrics_generator import LyricsGenerator
from xandly5.types.lyrics_model_enum import LyricsModelEnum

app = Flask(__name__)
api = Api(app)


# noinspection PyMethodMayBeStatic
class LyricsApi(Resource):

    def post(self):
        json_values = request.json

        # TODO AEO add validation
        model_id: LyricsModelEnum = json_values['model_id']
        seed_text: str = json_values['seed_text']
        word_count: int = json_values['word_count']
        word_group_count: int = json_values['word_group_count']

        generator = LyricsGenerator(model_id)
        lyrics = generator.generate_lyrics(seed_text=seed_text, word_count=word_count,
                                           word_group_count=word_group_count)

        response = make_response(lyrics, 200)
        response.mimetype = "text/plain"

        return response


api.add_resource(LyricsApi, '/lyrics-api')

if __name__ == '__main__':
    app.run(debug=False)  # enabling debug causes CUDNN errors

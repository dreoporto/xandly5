from flask import Flask, request, make_response, jsonify, render_template
from flask_marshmallow import Marshmallow, fields
from marshmallow import post_load
from flask_restful import Resource, Api

from xandly5.service.lyrics_generator import LyricsGenerator
from xandly5.types.lyrics_model_enum import LyricsModelEnum
from xandly5.types.validation_error import ValidationError
from xandly5.types.lyrics_section import LyricsSection

app = Flask(__name__)
api = Api(app)
ma = Marshmallow(app)


def make_error(status_code: int, message: str):
    response = jsonify({
        'status': status_code,
        'message': message
    })
    response.status_code = status_code
    return response


class LyricsSectionSchema(ma.Schema):
    section_id = fields.fields.Str(required=False, allow_none=True, load_default=True)
    section_type = fields.fields.Int()
    word_count = fields.fields.Int()
    word_group_count = fields.fields.Int()
    seed_text = fields.fields.Str()

    # noinspection PyUnusedLocal
    @post_load
    def make_section(self, data, **kwargs):
        return LyricsSection(**data)


sections_schema = LyricsSectionSchema(many=True)


# noinspection PyMethodMayBeStatic
class LyricsApi(Resource):

    def post(self):
        try:
            json_values = request.json
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

        except ValidationError as ve:
            return make_error(400, str(ve))


# noinspection PyMethodMayBeStatic
class StructuredLyricsApi(Resource):

    def post(self):
        try:
            json_values = request.json
            model_id: LyricsModelEnum = LyricsModelEnum(json_values['model_id'])
            independent_sections: bool = json_values['independent_sections']
            sections = json_values['lyrics_sections']
            lyrics_sections = sections_schema.load(sections, many=True, unknown='exclude')

            generator = LyricsGenerator(LyricsModelEnum(model_id))
            lyrics: str

            if independent_sections:
                lyrics = generator.generate_lyrics_from_independent_sections(lyrics_sections)
            else:
                lyrics = generator.generate_lyrics_from_sections(lyrics_sections)

            response = make_response(lyrics, 200)
            response.mimetype = "text/plain"

            return response

        except ValidationError as ve:
            return make_error(400, str(ve))


api.add_resource(LyricsApi, '/lyrics-api')
api.add_resource(StructuredLyricsApi, '/structured-lyrics-api')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')  # enabling debug causes CUDNN errors

# Xandly5 - NLP Lyrics Generator

## Applied AI for Lyrics Generation using Natural Language Processing (NLP) and Object-Oriented Python

> "The next big thing is a good song." -- Brian Epstein

Xandly5 is a lyrics generator powered by Natural Language Processing using the Keras and TensorFlow frameworks.  Deep Learning models are trained on separate collections of works to produce genre-specific output.

This project should *not* be seen as an attempt to replace creative artists.  This would be impossible and, more importantly, unwanted.  Xandly5 is an experimental tool to empower individuals by providing a springboard for ideas in songwriting and poetry.

## Public Domain Sources

To ensure we respect copyrights, all lyrics used to train models are from public domain works.  Xandly5 currently includes separate models for:

- Shakespeare's Sonnets
- Edgar Allan Poe's Complete Poetical Works

## Natural Language Processing with Keras and TensorFlow

### Keras and TensorFlow NLP Models

The NLP models in Xandly5 are currently Keras Sequential models with Embedding and Bidirectional Long Short-Term Memory (LSTM) layers.  This allows models to take starter text (aka seed text) specified by the user and predict the next set of words in a song or poem.  The LSTM layer is a Recurrent Neural Network layer that maintains memory, so that a word later in a song is influenced by earlier words.  The bidirectional capability enhances this functionality.

The modular nature of Xandly5 allows us to add additional models which may use other types of layers and Deep Learning technologies.

### Catalog class

The Catalog class uses the Keras Tokenizer to generate multiple N-gram sequences for each lyric line.  An example of a N-gram sequence from one line of poetry, with corresponding text:

```
LINE: once upon a midnight dreary

N-GRAM                          CORRESPONDING TEXT
[1362, 27]                      once upon
[1362, 27, 5]                   once upon a
[1362, 27, 5, 285]              once upon a midnight
[1362, 27, 5, 285, 1363]        once upon a midnight dreary

```

The tokenizer is fitted on all lyrics associated with the catalog, and then generates and pads multiple n-gram sequences for each line.  The tokenizer converts each word in the catalog to a number.  We use a separate `Catalog` class to reuse this functionality for both model training and later predictions, since the tokenizer *with consistent tokens* is required as part of both processes.

We train models to predict the last word for each sequence (ex: *upon* in the first example above, and *dreary* in the last).  Sequences are split into training and validation groups so that we can graph performance and identify issues with over- and under-fitting.  Xandly5 makes use of the PTMLib library, including Timer and Charts.

### Model Output

The output for each of these models are in lyrics that can be considered imperfect yet hopefully inspirational, and consistent with each model's genre thanks to the word selections/predictions.  Xandly5 output is not random, otherwise we would get different gibberish with each submission.  The same exact input to the same model with the same word count and grouping parameters will produce the same output, which can then be used a springboard for lyric ideas.  

Dolly Parton and her fellow songwriters will not be out of a job anytime soon.  Thank goodness.  Humanity needs them, now and always.  Especially now.

## Xandly5 Architecture 

<!--![Xandly5 Architecture Diagram](images/Xandly5-Architecture.svg)-->
<p align="center">
    <a href="images/Xandly5-Architecture.svg"><img src="images/Xandly5-Architecture.svg" width="60%"></a>
</p>

## AI ML Model

This module contains all code and files for generating models, and making word predictions.  Models are saved as H5 files.

- `Catalog`
    - used in model training
    - also used by the LyricsGenerator Service 
    - `catalog_items` stores all lyrics lines for a corpus (collection of works, i.e. Sonnets)
    - `tokenizer` is used in both model training and prediction stages
    - `generate_lyrics_text` - generates lyrics using the associated tokenizer
- `LyricsModel` trains models and saves them as H5 files; specific child models include:
    - `SharespearSonnetModel`
    - `PoePoemModel`
    - Each child model has an asscoated `*_config.json` with hyperparameter settings
    - Each child model can be executed to save your own custom H5 models.  
    - We recommend downloading our saved H5 models as part of the setup instructions below.
- `LyricsFormatter` is used to generate formatted lyrics that are more readable
- `saved_models` folder - models are saved/stored here in H5 format
- `lyrics_files` folder - source lyrics files in TXT format

## Service

### LyricsGenerator

The `LyricsGenerator` class provides logic and validations for generating lyrics using pre-trained models:  

`generate_lyrics` - This method generates lyrics, using the specified seed text.  The `word_count` parameter specifies the total number of words to generate.  The `word_group_count` parameter will add a comma or blank line, alternately, after the number of specified words.

`generate_lyrics_from_sections` - One feature that makes Xandly5 unique is the ability to generate lyrics for a specified song structure.  This is accomplished using the `LyricsSection` type. A user can specify a list of song sections, each with their own seed text, word count, etc.  

Structured Lyrics Example:

```
sections: List[LyricsSection] = [
    LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                    seed_text='a dreary  midnight bird and here i heard'),
    LyricsSection(section_type=SectionTypeEnum.CHORUS, word_group_count=4, word_count=16,
                    seed_text='said he art too seas for totter into'),
    LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                    seed_text='tone of his eyes of night litten have'),
    LyricsSection(section_type=SectionTypeEnum.CHORUS, word_group_count=4, word_count=16,
                    seed_text='said he art too seas for totter into'),
```

Generate text for each section is dependent on the `seed_text` value, and is dependent on text from prior sections thanks to LSTM.

An alternate `generate_lyrics_from_independent_sections` method is also available, allow text for each section to be generated regardless of the text in other sections.

### Unit Tests

A `tests` folder contains unit tests and related files to ensure text is generated consistently.  You will see different results if you generate your own H5 files rather than download the ones we provide as part of the Install process below.

## Web - REST API and UI

- Flask REST
    - API
    - Examples
- UI - HTML5, Bootstrap, CSS, jQuery
    - jQuery... keeping it simple (for now)

## Types

## Install / Setup

## Next Steps

## Conclusion
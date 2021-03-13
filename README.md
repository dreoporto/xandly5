# Xandly5 - NLP Lyrics Generator

## Applied AI for Lyrics Generation using Natural Language Processing (NLP) and Object-Oriented Python

> "The next big thing is a good song." 
> - Brian Epstein

Xandly5 is a lyrics generator powered by Natural Language Processing using the Keras and TensorFlow frameworks.  Deep Learning models are trained on separate collections of works to produce genre-specific output.

This project should *not* be seen as an attempt to replace creative artists.  This would be impossible and, more importantly, unwanted.  Xandly5 is an experimental tool to empower individuals by providing a springboard for ideas in songwriting and poetry.

## Public Domain Sources

To ensure we respect copyrights, all lyrics used to train models are from public domain works.  Xandly5 currently includes separate models for:

- Shakespeare's Sonnets
- Edgar Allan Poe's Complete Poetical Works

## Natural Language Processing with Keras and TensorFlow

### Keras and TensorFlow NLP Models

The NLP models in Xandly5 are currently Keras Sequential models with Embedding and Bidirectional Long Short-Term Memory (LSTM) layers.  This allows models to take starter text specified by the user and predict the next set of words.  The LSTM layer is a Recurrent Neural Network layer that maintains memory, so that a word later in a song is influenced by earlier words.  The bidirectional capability enhances this functionality.

The modular design of Xandly5 allows us to add additional models, which can leverage other Deep Learning layers, techniques, and frameworks.

### Catalog class

The Keras Tokenizer is used to create multiple N-gram sequences for each lyric line.  An example of a N-gram sequence from one line of poetry, with corresponding text:

```
LINE: once upon a midnight dreary

N-GRAM                          CORRESPONDING TEXT
[1362, 27]                      once upon
[1362, 27, 5]                   once upon a
[1362, 27, 5, 285]              once upon a midnight
[1362, 27, 5, 285, 1363]        once upon a midnight dreary

```

The tokenizer is fitted on all lyrics associated with the catalog, and then generates and pads multiple n-gram sequences for each line.  The tokenizer converts each word in the catalog to a number.  

The `Catalog` class in Xandly5 allows us to reuse this functionality for model training, and *later word predictions* for end users, since a tokenizer with consistent vocabulary and settings is required as part of both processes.

We train models to predict the last word for each sequence (ex: *upon* in the first example above, and *dreary* in the last).  Sequences are split into training and validation groups so that we can graph performance and identify issues with over- and under-fitting.  Xandly5 makes use of the `Stopwatch` and `charts` modules from the [PTMLib](https://github.com/dreoporto/ptmlib) library.

### Model Output

The lyrics produced by these models can be considered imperfect yet hopefully inspirational.  They are consistent with each model genre thanks to the word predictions; sometimes they rhyme.  Xandly5 output is not random, since this would produce different results with each submission.  The same exact text input to the same model, with the same word count and grouping parameters, will produce the same output, which can then be used as a springboard for ideas.  

Dolly Parton and her fellow songwriters will not be out of a job anytime soon.  Thank goodness.  Humanity needs them, now and always.  Especially now.

## Xandly5 Architecture 

<!--TODO change link when repo is public-->
<p align="center">
    <a href="images/Xandly5-Architecture.svg"><img src="images/Xandly5-Architecture.svg" width="90%"></a>
</p>

## `ai_ml_model`

This module contains all code and files for training and saving models, and making word predictions.

### `LyricsModel` - Trains models and saves them as H5 files

- Specific child models currently included:
    - `ShakespeareSonnetModel`
    - `PoePoemModel`
- Each child model has an associated `*_config.json` with hyperparameter settings
- Each child model can be executed to save your own custom H5 models
- We recommend downloading the H5 models per the setup instructions below

### `Catalog`

- Used for both model training, and prediction (via the `LyricsGenerator` service)
- `catalog_items` stores all lyrics for a corpus (i.e. collection of works)
- `generate_lyrics_text` - creates lyrics using the Catalog's associated tokenizer and related properties

### Additional Items
- `LyricsFormatter` - used to produce formatted lyrics that are more readable
- `saved_models` folder - models are saved/stored here in H5 format
- `lyrics_files` folder - source lyrics files in TXT format

## `service`

### `LyricsGenerator` - provides logic and validations for generating lyrics using pre-trained models

- `generate_lyrics` - This method creates lyrics, using the specified seed text
- `word_count` - Parameter specifies the total number of words to generate
- `word_group_count` - Parameter controls the addition of commas or blank lines, alternately, after the number of specified words

#### Song Structure: the LyricsGenerator and LyricsSection classes

`generate_lyrics_from_sections` - One feature that makes Xandly5 unique is the ability to produce lyrics with a specified song structure.  This is accomplished using this method of the `LyricsGenerator` class, along with the `LyricsSection` type. A user can specify a list of song sections, each with their own seed text, word count, etc.  

Structured Lyrics Example:

```python
sections: List[LyricsSection] = [
    LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                    seed_text='a dreary midnight bird'),
    LyricsSection(section_type=SectionTypeEnum.CHORUS, word_group_count=4, word_count=16,
                    seed_text='said he art too'),
    LyricsSection(section_type=SectionTypeEnum.VERSE, word_group_count=4, word_count=32,
                    seed_text='tone of his eyes')
    ...
```

The generated text for each section is dependent on the `seed_text` value, and text from *prior sections*, thanks to LSTM.

`generate_lyrics_from_independent_sections` - This alternative method creates text for each section without being influenced by the text in other sections.

### Unit Tests

A `tests` folder contains unit tests and related files to ensure text is generated consistently.  

IMPORTANT NOTE: You will see different results if you create your own H5 files rather than download the ones we provide in the Install process below.

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
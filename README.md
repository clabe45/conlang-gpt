# Conlang GPT

Conlang GPT is a command line tool for creating, modifying and using constructed languages, powered by ChatGPT

## Features

> :warning: This tool is in early development. It is not ready for production use. Regardless, anyone who's generating fake languages in a business context should carefully consider their life choices, but this is fine because it's open source.

| Feature | Status |
| --- | --- |
| Automatically generate specifications for written or spoken languages | :sparkles: Alpha |
| Automatically or manually improve language specs iteratively | :sparkles: Alpha |
| Translate any text to or from the generated languages | :sparkles: Alpha |
| Generate vocabulary lazily | :sparkles: Alpha |
| Automatically update vocabulary when language specs change | :sparkles: Alpha |
| Support for all of OpenAI's chat models | :sparkles: Alpha |

## Installation

```
pip install conlang-gpt
```

## Anatomy of a Language

Languages are represented as two files:
- **Guide**: The purpose of the language guide is to describe how to use the language, including rules related to grammar and phonetics.
- **Dictionary**: The dictionary contains the vocabulary for the language. It is built up lazily as more and more text is translated.

## Commands

### Overview

Conlang GPT provides several commands:

```
$ conlang --help
Usage: conlang [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  create     Create a constructed language.
  improve    Automatically improve the language.
  modify     Make specific changes to the language.
  translate  Translate text to or from a constructed language.
```

Before running any of them, set the `OPENAI_API_KEY` environment variable (keep the space in front to exclude the command from your history):

```
 export OPENAI_API_KEY=sk...
```

### `conlang create`

Creates the first draft of a guide for a new language and writes it to a file.

```
$ conlang create --help
Usage: conlang create [OPTIONS]

  Create a constructed language.

Options:
  --design-goals TEXT
  --guide TEXT
  --model TEXT         OpenAI model to use. Defaults to gpt-3.5-turbo.
  --help               Show this message and exit.
```

### `conlang improve`

Attempts to automatically improve the language guide and dictionary. The resulting guide and dictionary are saved to the original input files.

```
$ conlang improve --help
Usage: conlang improve [OPTIONS]

  Automatically improve the language.

Options:
  --guide TEXT
  --dictionary TEXT             Enter the filename of the dictionary to use in
                                'example' mode.
  --max-iterations INTEGER      Max number of revisions to perform. Defaults
                                to 5.
  --similarity-threshold FLOAT  Maximum similarity between two words to be
                                considered the same. Defaults to 0.98.
  --model TEXT                  OpenAI model to use. Defaults to
                                gpt-3.5-turbo.
  --embeddings-model TEXT       OpenAI model to use for word embeddings in
                                'example' mode. Defaults to text-embedding-
                                ada-002.
  --help                        Show this message and exit.
```

### `conlang modify`

Makes a specific change to the language and updates the guide and dictionary.

```
$ conlang modify --help
Usage: conlang modify [OPTIONS]

  Make specific changes to the language.

Options:
  --guide TEXT
  --dictionary TEXT             Enter the filename of the dictionary to use in
                                'example' mode.
  --changes TEXT
  --similarity-threshold FLOAT  Maximum similarity between two words to be
                                considered the same. Defaults to 0.98.
  --model TEXT                  OpenAI model to use. Defaults to
                                gpt-3.5-turbo.
  --embeddings-model TEXT       OpenAI model to use for word embeddings in
                                'example' mode. Defaults to text-embedding-
                                ada-002.
  --help                        Show this message and exit.
```

### `conlang translate`

Translates text between any language ChatGPT was trained on to and from the conlang. The language of the input text is automatically detected and used to dermine which language to translate to. If any problems are encountered while translating, the guide and dictionary will be repeatedly fixed until `--max-improvements` is reached.

```
$ conlang translate --help
Usage: conlang translate [OPTIONS]

  Translate text to or from a constructed language.

Options:
  --guide TEXT
  --dictionary TEXT
  --text TEXT
  --max-improvements INTEGER    Max number of relevant improvements to make to
                                the guide and dictionary. Defaults to 5.
  --similarity-threshold FLOAT  Maximum similarity between two words to be
                                considered the same. Defaults to 0.98.
  --model TEXT                  OpenAI model to use
  --embedding-model TEXT        OpenAI model to use for word embeddings
  --help                        Show this message and exit.
```

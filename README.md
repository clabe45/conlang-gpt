# Conlang GPT

Conlang GPT is a command line tool for creating, modifying and using constructed languages, powered by ChatGPT

## Features

| Feature | Status |
| --- | --- |
| Automatically generate specifications for written or spoken languages | :white_check_mark: Stable |
| Automatically or manually improve languages iteratively | :white_check_mark: Stable |
| Generate vocabulary | :sparkles: Experimental |
| Translate any text to or from the generated languages | :white_check_mark: Stable |
| Support for all of OpenAI's chat models | :white_check_mark: Stable |

## Installation

```
pip install conlang-gpt
```

## Commands

### Overview

Conlang GPT provides several commands:

```
$ conlang --help
Usage: conlang [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  dictionary
  guide
  text
```

Before running any of them, set the `OPENAI_API_KEY` environment variable (keep the space in front to exclude the command from your history):

```
 export OPENAI_API_KEY=sk...
```

### `conlang guide`

#### Overview

```
$ conlang guide --help
Usage: conlang guide [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  create   Create a constructed language.
  improve  Automatically improve the language.
  modify   Make specific changes to the language.
```

#### `conlang guide create`

Creates a guide for a new language and writes it to a file.

```
$ conlang guide create --help
Usage: conlang guide create [OPTIONS]

  Create a constructed language.

Options:
  --design-goals TEXT
  --output TEXT
  --model TEXT         OpenAI model to use. Defaults to gpt-3.5-turbo-16k.
  --help               Show this message and exit.
```

#### `conlang guide improve`

Attempts to automatically improve the language and saves the resulting guide to a file. Supports two modes - `basic` and `example`. In basic mode, a random flaw with the language is identified and fixed. In example mode, a random English sentence is generated, translated to the conlang and used to identify and fix a problem with the language.

```
$ conlang guide improve --help
Usage: conlang guide improve [OPTIONS]

  Automatically improve the language.

Options:
  --input TEXT
  --output TEXT
  --mode [basic|example]  Mode to use. Defaults to basic. Set to the
                          experimental 'example' mode to include a new random
                          translation in each revision.
  -n INTEGER              Number of revisions to perform. Defaults to 1.
  --model TEXT            OpenAI model to use. Defaults to gpt-3.5-turbo-16k.
  --help                  Show this message and exit.
```

#### `conlang guide modify`

Makes a specific change to the language and updates the guide.

```
$ conlang guide modify --help
Usage: conlang guide modify [OPTIONS]

  Make specific changes to the language.

Options:
  --input TEXT
  --output TEXT
  --changes TEXT
  --model TEXT         OpenAI model to use. Defaults to gpt-3.5-turbo-16k.
  --help               Show this message and exit.
```

### `conlang text`

#### Overview

```
$ conlang text --help
Usage: conlang text [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  translate  Translate text to or from a constructed language.
```

#### `conlang text translate`

Translates text between any language ChatGPT was trained on to and the conlang. The language of the input text is automatically detected and used to dermine which language to translate to.

```
$ conlang translate --help
Usage: conlang translate [OPTIONS]

  Translate text to or from a constructed language.

Options:
  --guide TEXT
  --text TEXT
  --model TEXT  OpenAI model to use
  --help        Show this message and exit.
```

### `conlang dictionary` (experimental)

#### Overview

```
$ conlang dictionary --help
Usage: conlang text [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  translate  Translate text to or from a constructed language.
```

#### `conlang dictionary add`

```
$ conlang dictionary add --help
Usage: conlang dictionary add [OPTIONS]

  Generate words in the language (experimental).

Options:
  --guide TEXT
  --output TEXT
  -n INTEGER     Number of words to generate. Defaults to 15.
  --model TEXT   OpenAI model to use. Defaults to gpt-3.5-turbo-16k.
  --help         Show this message and exit.
```

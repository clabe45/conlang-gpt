# Conlang GPT

Conlang GPT is a command line tool for creating, modifying and using constructed languages, powered by ChatGPT

## Features

> :warning: This tool is in early development. It is not ready for production use.

| Feature | Status |
| --- | --- |
| Automatically generate specifications for written or spoken languages | :sparkles: Alpha |
| Automatically or manually improve languages iteratively | :sparkles: Alpha |
| Generate vocabulary lazily | :sparkles: Alpha |
| Translate any text to or from the generated languages | :sparkles: Alpha |
| Support for all of OpenAI's chat models | :sparkles: Alpha |

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

Creates a guide for a new language and writes it to a file.

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

Attempts to automatically improve the language and saves the resulting guide to a file. Supports two modes - `simple` and `example`. In simple mode, a random flaw with the language is identified and fixed. In example mode, a random English sentence is generated, translated to the conlang and used to identify and fix a problem with the language.

```
$ conlang improve --help
Usage: conlang improve [OPTIONS]

  Automatically improve the language.

Options:
  --guide TEXT
  --dictionary TEXT        Enter the filename of the dictionary to use in
                           'example' mode.
  --text TEXT
  -n INTEGER               Number of revisions to perform. Defaults to 1.
  --model TEXT             OpenAI model to use. Defaults to gpt-3.5-turbo.
  --embeddings-model TEXT  OpenAI model to use for word embeddings in
                           'example' mode. Defaults to text-embedding-ada-002.
  --help                   Show this message and exit.
```

### `conlang modify`

Makes a specific change to the language and updates the guide.

```
$ conlang modify --help
Usage: conlang modify [OPTIONS]

  Make specific changes to the language.

Options:
  --guide TEXT
  --changes TEXT
  --model TEXT    OpenAI model to use. Defaults to gpt-3.5-turbo.
  --help          Show this message and exit.
```

### `conlang translate`

Translates text between any language ChatGPT was trained on to and the conlang. The language of the input text is automatically detected and used to dermine which language to translate to.

```
$ conlang translate --help
Usage: conlang translate [OPTIONS]

  Translate text to or from a constructed language.

Options:
  --guide TEXT
  --dictionary TEXT
  --text TEXT
  --model TEXT            OpenAI model to use
  --embedding-model TEXT  OpenAI model to use for word embeddings
  --help                  Show this message and exit.
```

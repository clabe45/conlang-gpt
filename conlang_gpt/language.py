import csv
import io
from openai.embeddings_utils import cosine_similarity
import os
import pickle

import click

from .openai import complete_chat, get_embedding


class LanguageError(Exception):
    """Abstract base class for errors related to a constructed language."""

    pass


class DictionaryError(LanguageError):
    """Exception raised when there is an error with a dictionary."""

    pass


class CreateDictionaryError(DictionaryError):
    """Exception raised when a dictionary cannot be created."""

    pass


class ImproveDictionaryError(DictionaryError):
    """Exception raised when the dictionary cannot be updated."""

    pass


class NoDictionaryError(DictionaryError):
    """
    Exception raised when a dictionary is not found in a response from ChatGPT.
    """

    pass


class InvalidDictionaryError(DictionaryError):
    """Exception raised when a dictionary is invalid."""

    pass


def _get_embeddings(text, embeddings_model):
    """Retrieve and cache the embeddings for some text."""

    # Load the cached word embeddings
    if os.path.exists(f".conlang/cache/embeddings/{embeddings_model}.pkl"):
        with open(f".conlang/cache/embeddings/{embeddings_model}.pkl", "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}

    # Calculate the embeddings if they are not cached
    if text not in embeddings:
        embeddings[text] = get_embedding(text, embeddings_model)

        # Cache the embeddings
        if not os.path.exists(".conlang/cache/embeddings"):
            os.makedirs(".conlang/cache/embeddings")
        with open(f".conlang/cache/embeddings/{embeddings_model}.pkl", "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings[text]


def _get_related_words(text, dictionary, embeddings_model):
    """Get the most related words from the dictionary."""

    click.echo(
        click.style(f"Getting the most relevant words from the dictionary...", dim=True)
    )

    # The longer the text, the more words we want to return
    max_words = min(len(dictionary), int(len(text) / 2.5))

    # Get the embeddings for the text
    text_embeddings = _get_embeddings(text, embeddings_model)

    # Calculate the cosine similarity between the text and each word in the dictionary (both the word and the English translation)
    word_similarities = {}
    for word, translation in dictionary.items():
        # Calculate the cosine similarity between the text and the word
        word_embeddings = _get_embeddings(word, embeddings_model)
        word_simularity = cosine_similarity(text_embeddings, word_embeddings)

        # Calculate the cosine similarity between the text and the translation
        translation_embeddings = _get_embeddings(translation, embeddings_model)
        translation_simularity = cosine_similarity(
            text_embeddings, translation_embeddings
        )

        simularity = max(word_simularity, translation_simularity)

        # Only include words with a simularity greater than 0.75
        if simularity > 0.75:
            word_similarities[word] = simularity

    # Sort the words by similarity
    related_words = sorted(word_similarities, key=word_similarities.get, reverse=True)

    # Return the most similar words
    if len(related_words) > max_words:
        return related_words[:max_words]
    else:
        return related_words


def _parse_dictionary(text, similarity_threshold, embeddings_model):
    # Extract the CSV document, unless the text is just a message
    document = None
    if "\n\n" in text:
        click.echo(text)
        for paragraph in text.split("\n\n"):
            if paragraph.startswith("Conlang,English"):
                document = paragraph
                break
        else:
            raise NoDictionaryError(text)
    else:
        document = text

    # Parse the CSV document
    reader = csv.reader(document.splitlines())
    header = next(reader)
    header = [column.strip() for column in header]

    # Parse the words
    dictionary = {}
    for row in reader:
        # Extract the word and translation
        if len(row) != 2:
            raise InvalidDictionaryError(
                f"Invalid response. Expected row to have two columns: Word and Translation. Received: {row}"
            )
        word, translation = row

        # If the word starts with a number (e.g., "1. hello"), remove the number
        if "." in word:
            first, rest = word.split(".", 1)
            if first.isdigit():
                word = rest.strip()

        # Remove any trailing whitespace
        word = word.strip()
        translation = translation.strip()

        # Add the word to the dictionary
        dictionary[word] = translation

    # Remove similar words
    dictionary = reduce_dictionary(dictionary, similarity_threshold, embeddings_model)

    return dictionary


def translate_text(text, language_guide, dictionary, model, embeddings_model):
    """Translate text into a constructed language."""

    # Get the most related words from the dictionary
    related_words = _get_related_words(text, dictionary, embeddings_model)

    # Translate the text
    click.echo(click.style(f"Translating text using {model}...", dim=True))
    if related_words:
        formatted_related_words = "\n".join(
            [f"- {word}: {dictionary[word]}" for word in related_words]
        )
        click.echo(
            click.style(f"Most related words:\n\n{formatted_related_words}", dim=True)
        )
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the text below from or into the following constructed language. Explain how you arrived at the translation. Only use words found in either the guide or the list below.\n\nLanguage guide:\n\n{language_guide}\n\nPotentially-related words:\n\n{formatted_related_words}\n\nText to translate:\n\n{text}",
                }
            ],
        )
        translation = chat_completion["choices"][0]["message"]["content"]

    else:
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the text below from or into the following constructed language. Explain how you arrived at the translation. Only use words found in the guide.\n\nNo relevant words from dictionary found.\n\nLanguage guide:\n\n{language_guide}\n\nText to translate:\n\n{text}",
                }
            ],
        )
        translation = chat_completion["choices"][0]["message"]["content"]

    return translation


def generate_english_text(model):
    click.echo(
        click.style(f"Generating random English text using {model}...", dim=True)
    )
    chat_completion = complete_chat(
        model=model,
        temperature=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are a writing assistant who likes to write about different topics.",
            },
            {"role": "user", "content": "Please generate a random English sentence."},
        ],
    )

    english_text = chat_completion["choices"][0]["message"]["content"]
    return english_text


def improve_language(guide, dictionary, model, embeddings_model, text=None):
    click.echo(click.style(f"Improving language using {model}...", dim=True))

    if text is None:
        # Identify problems with the language
        chat_completion = complete_chat(
            model=model,
            temperature=0.5,
            presence_penalty=0.5,
            messages=[
                {
                    "role": "user",
                    "content": f'If the language outlined below has any flaws, contradictions or points of confusion, please identify one and provide specific, detailed, actionable steps to fix it. Otherwise, respond with "No problem found" instead of the csv document. Note that a dictionary is provided separately.\n\nLanguage guide:\n\n{guide}',
                }
            ],
        )
        revisions = chat_completion["choices"][0]["message"]["content"]

    else:
        # Attempt to translate the provided text
        translation = translate_text(text, guide, dictionary, model, embeddings_model)
        click.echo(f"Translated text:\n\n{translation}\n")

        # Identify problems with the language using the translated text as an example/reference
        chat_completion = complete_chat(
            model=model,
            temperature=0.1,
            # TODO: Replace with `presence_penalty=0.1`
            frequency_penalty=0.1,
            messages=[
                {
                    "role": "user",
                    "content": f'If the language outlined below has any flaws, contradictions or points of confusion, please identify one and provide specific, detailed, actionable steps to fix it. Otherwise, respond with "No problem found". I included a sample translation to give you more context.\n\nLanguage guide:\n\n{guide}\n\nOriginal text: {text}\n\nTranslated text: {translation}',
                }
            ],
        )
        revisions = chat_completion["choices"][0]["message"]["content"]

    # Check if any problems were found
    if "No problem found" in revisions or "No problems found" in revisions:
        click.echo("No problems found.")
        return None

    click.echo(f"Change:\n\n{revisions}\n")

    # Rewrite the language guide
    chat_completion = complete_chat(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": f"Improve the following constructed language to address the problem described below. Your response should be a reference sheet describing the new language's rules. Assume the reader did not read the original reference sheet and that they have no prior experience using the language. Note that a dictionary is provided separately.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{revisions}",
            }
        ],
    )
    improved_guide = chat_completion["choices"][0]["message"]["content"]

    return improved_guide


def generate_language(design_goals, model):
    """Generate a constructed language."""

    click.echo(f"Generating language using {model}...")
    chat_completion = complete_chat(
        model=model,
        temperature=0.9,
        presence_penalty=0.5,
        messages=[
            {
                "role": "user",
                "content": f"Create a constructed language with the following design goals:\n\n{design_goals}\n\nYour response should be a reference sheet including all the language's rules. Assume the reader has no prior experience using the language.",
            }
        ],
    )
    guide = chat_completion["choices"][0]["message"]["content"]
    click.echo(f"Initial draft:\n\n{guide}\n")

    return guide


def modify_language(guide, changes, model):
    """Apply specified changes to a constructed language."""

    click.echo(click.style(f"Modifying language using {model}...", dim=True))
    chat_completion = complete_chat(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": f"Make the following changes to the constructed language outlined below. Your response should be a reference sheet describing the new language's rules. Assume the reader did not read the original reference sheet and that they have no prior experience using the language. Note that a dictionary is provided separately.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{changes}",
            }
        ],
    )
    improved_guide = chat_completion["choices"][0]["message"]["content"]

    return improved_guide


def reduce_dictionary(words, similarity_threshold, embeddings_model):
    """Remove similar words from a dictionary."""

    click.echo(
        click.style(f"Removing similar words using {embeddings_model}...", dim=True)
    )

    # Retrieve the embeddings for each word
    word_embeddings = {word: _get_embeddings(word, embeddings_model) for word in words}

    # Remove similar words
    words_to_remove = set()
    for word_a, embedding_a in word_embeddings.items():
        for word_b, embedding_b in word_embeddings.items():
            if (
                word_a != word_b
                and cosine_similarity(embedding_a, embedding_b) > similarity_threshold
            ):
                click.echo(
                    click.style(
                        f"Removed {word_b} because it is similar to {word_a}.", dim=True
                    )
                )
                words_to_remove.add(word_b)

    # Remove the similar words from the dictionary
    for word in words_to_remove:
        words.pop(word)

    return words


def create_dictionary_for_text(
    guide, text, existing_dictionary, similarity_threshold, model, embeddings_model
) -> dict:
    """Generate words for a constructed language."""

    click.echo(click.style(f"Generating words using {model}...", dim=True))

    # Get related words from the existing dictionary
    related_words = _get_related_words(text, existing_dictionary, embeddings_model)

    # Format the related words as a CSV document
    mutable_formatted_related_words = io.StringIO()
    writer = csv.writer(mutable_formatted_related_words)
    writer.writerow(["Conlang", "English"])
    for word in related_words:
        writer.writerow([word, existing_dictionary[word]])
    formatted_related_words = mutable_formatted_related_words.getvalue()

    # Generate words
    if len(related_words) > 0:
        click.echo(f"Related words:\n\n{formatted_related_words}\n")
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Create any missing words required to translate the following text into the constructed language outlined below. Your response should be a CSV document with two columns: Conlang and English. Each row should have exactly two cells.\n\nLanguage guide:\n\n{guide}\n\nText to translate (either from or to the conlang):\n\n{text}\n\nExisting words that could be related:\n\n{formatted_related_words}",
                }
            ],
        )
        response = chat_completion["choices"][0]["message"]["content"]

    else:
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Create all the words required to translate the following text into the constructed language outlined below. Your response should be a CSV document with two columns: Conlang and English. Each row should have exactly two cells.\n\nLanguage guide:\n\n{guide}\n\nText to translate (either from or to the conlang):\n\n{text}\n\nNo existing words are related to the text.",
                }
            ],
        )
        response = chat_completion["choices"][0]["message"]["content"]

    # Parse the generated words
    try:
        words = _parse_dictionary(response, similarity_threshold, embeddings_model)
    except NoDictionaryError as e:
        raise CreateDictionaryError from e

    return words


def improve_dictionary(
    dictionary, guide, similarity_threshold, model, embeddings_model, batch_size=25
):
    """Update the dictionary to match the guide by focusing on updating the words themselves instead of their translations."""

    click.echo(click.style(f"Improving dictionary using {model}...", dim=True))

    # Get the words to improve
    words_to_improve = list(dictionary.keys())

    # Improve the words
    for i in range(0, len(words_to_improve), batch_size):
        # Dump the batch to a csv string
        mutable_batch_string = io.StringIO()
        writer = csv.writer(mutable_batch_string)
        writer.writerow(["Conlang", "English"])
        for word in words_to_improve[i : i + batch_size]:
            writer.writerow([word, dictionary[word]])
        formatted_batch = mutable_batch_string.getvalue()

        # Improve the words
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f'Ensure that the following words are correctly translated into the constructed language outlined below. If any of the words do not adhere to the guide below, update or remove them as you see fit. Your response should be a CSV document with two columns: Conlang and English. Each row should have exactly two cells. If the word list below is correct and complete, respond with "No problems found".\n\nLanguage guide:\n\n{guide}\n\nWords to improve:\n\n{formatted_batch}',
                }
            ],
        )
        response = chat_completion["choices"][0]["message"]["content"]

        # Parse the response
        if "No problems found" in response:
            continue

        try:
            improved_batch = _parse_dictionary(
                response, similarity_threshold, embeddings_model
            )
        except NoDictionaryError as e:
            raise ImproveDictionaryError from e

        # Remove the old batch from the dictionary
        for word in words_to_improve[i : i + batch_size]:
            del dictionary[word]

        # Add the improved batch to the dictionary
        dictionary = merge_dictionaries(
            dictionary, improved_batch, similarity_threshold, embeddings_model
        )

    return dictionary


def merge_dictionaries(a, b, similarity_threshold, embeddings_model):
    """Merge two vocabulary dictionaries, removing similar words."""

    # Retrieve the embeddings for each word
    a_embeddings = {word: _get_embeddings(word, embeddings_model) for word in a.keys()}
    b_embeddings = {word: _get_embeddings(word, embeddings_model) for word in b.keys()}

    # Calculate the cosine similarity between each pair of words
    a = dict(a)
    b = dict(b)
    similarities = {}
    for a_word, a_embedding in a_embeddings.items():
        for b_word, b_embedding in b_embeddings.items():
            similarities[(a_word, b_word)] = cosine_similarity(a_embedding, b_embedding)

    # Remove words that are too similar. Prefer shorter words.
    for (a_word, b_word), similarity in similarities.items():
        # Skip words that have already been removed
        if a_word not in a or b_word not in b:
            continue

        if similarity > similarity_threshold:
            if len(b_word) < len(a_word):
                click.echo(
                    click.style(
                        f"Removing {a_word} because it is too similar to {b_word}.",
                        dim=True,
                    )
                )
                del a[a_word]
            else:
                click.echo(
                    click.style(
                        f"Removing {b_word} because it is too similar to {a_word}.",
                        dim=True,
                    )
                )
                del b[b_word]

    # Merge the dictionaries
    merged = {**a, **b}

    return merged


def load_dictionary(dictionary_path):
    # Load the dictionary
    if os.path.exists(dictionary_path):
        with open(dictionary_path, "r") as file:
            reader = csv.reader(file)

            # Skip the header row
            next(reader)

            # Load the dictionary
            dictionary = {row[0]: row[1] for row in reader}
    else:
        dictionary = {}

    return dictionary


def save_dictionary(dictionary, dictionary_path):
    # Save the dictionary in alphabetical order
    with open(dictionary_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Word", "Translation"])
        for word in sorted(dictionary.keys()):
            writer.writerow([word, dictionary[word]])

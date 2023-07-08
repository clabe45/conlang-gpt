import csv
from openai.embeddings_utils import cosine_similarity
import os
import pickle

import click

from .openai import complete_chat, get_embedding


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

def _get_related_words(text, dictionary, embeddings_model, max_words):
    """Get the most related words from the dictionary."""

    click.echo(click.style(f"Getting the most relevant words from the dictionary...", dim=True))

    # Validate the number of words to return
    if max_words > len(dictionary):
        raise ValueError("The number of words to return cannot be greater than the number of words in the dictionary.")

    # Get the embeddings for the text
    text_embeddings = _get_embeddings(text, embeddings_model)

    # Calculate the cosine similarity between the text and each word in the dictionary
    word_similarities = {}
    for word in dictionary:
        word_embeddings = _get_embeddings(word, embeddings_model)
        simularity = cosine_similarity(text_embeddings, word_embeddings)

        # Only include words with a simularity greater than 0.85
        if simularity > 0.85:
            word_similarities[word] = simularity

    # Sort the words by similarity
    related_words = sorted(word_similarities, key=word_similarities.get, reverse=True)

    # Return the most similar words
    if len(related_words) > max_words:
        return related_words[:max_words]
    else:
        return related_words

def translate_text(text, language_guide, dictionary, model, embeddings_model):
    """Translate text into a constructed language."""

    # Get the most related words from the dictionary
    # The longer the text, the more words we want to return
    max_words = min(10, int(len(text) / 2.5))
    related_words = _get_related_words(text, dictionary, embeddings_model, max_words)
    formatted_related_words = "\n".join([f"- {word}: {dictionary[word]}" for word in related_words])
    click.echo(click.style(f"Most related words:\n\n{formatted_related_words}", dim=True))

    # Translate the text
    click.echo(click.style(f"Translating text using {model}...", dim=True))
    chat_completion = complete_chat(
        model=model,
        messages=[{"role": "user", "content": f"Translate the text below from or into the following constructed language. Explain how you arrived at the translation.\n\nLanguage guide:\n\n{language_guide}\n\nPotentially-related words:\n\n{formatted_related_words}\n\nText to translate:\n\n{text}"}]
    )

    translation = chat_completion['choices'][0]['message']['content']
    return translation

def generate_english_text(model):
    click.echo(click.style(f"Generating random English text using {model}...", dim=True))
    chat_completion = complete_chat(
        model=model,
        temperature=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        messages=[
            {"role": "system", "content": "You are a writing assistant who likes to write about different topics."},
            {"role": "user", "content": "Please generate a random English sentence."}
        ]
    )

    english_text = chat_completion['choices'][0]['message']['content']
    return english_text

def improve_language(guide, model, mode):
    click.echo(click.style(f"Improving language using {model}...", dim=True))

    if mode == "simple":
        # Identify problems with the language
        chat_completion = complete_chat(
            model=model,
            temperature=0.5,
            presence_penalty=0.5,
            messages=[{"role": "user", "content": f"Please identify one flaw, contradiction or point of confusion with the language outlined below along with specific, detailed, actionable steps to fix it.\n\nLanguage guide:\n\n{guide}"}]
        )
        revisions = chat_completion['choices'][0]['message']['content']

    elif mode == "example":
        # Attempt to translate a random English sentence
        english_text = generate_english_text(model)
        translation = translate_text(english_text, guide, model)
        click.echo(f"Sample English text:\n\n{english_text}\n")
        click.echo(f"Translated text:\n\n{translation}\n")

        # Identify problems with the language using the translated text as an example/reference
        chat_completion = complete_chat(
            model=model,
            temperature=0.1,
            messages=[{"role": "user", "content": f"Please identify one flaw or point of confusion with the language outlined below along with specific, detailed, actionable steps to fix it. I included a sample translation to give you more context.\n\nLanguage guide:\n\n{guide}\n\nSample English text: {english_text}\n\nTranslated text: {translation}"}]
        )
        revisions = chat_completion['choices'][0]['message']['content']
    else:
        raise Exception("Invalid mode. Please set the mode to 'simple' or 'example'.")

    click.echo(f"Change:\n\n{revisions}\n")

    # Rewrite the language guide
    chat_completion = complete_chat(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "user", "content": f"Improve the following constructed language to address the problem described below. Your response should be an exhaustive reference sheet describing the new language. Assume the reader did not read the original reference sheet and that they have no prior experience using the language.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{revisions}"}
        ]
    )
    improved_guide = chat_completion['choices'][0]['message']['content']

    return improved_guide

def generate_language(design_goals, model):
    """Generate a constructed language."""

    click.echo(f"Generating language using {model}...")
    chat_completion = complete_chat(
        model=model,
        temperature=0.9,
        messages=[
            {"role": "user", "content": f"Create a constructed language with the following design goals:\n\n{design_goals}\n\nYour response should be an exhaustive reference sheet including all the information that is needed to use the language. Assume the reader has no prior experience using the language."}
        ],
    )
    guide = chat_completion['choices'][0]['message']['content']
    click.echo(f"Initial draft:\n\n{guide}\n")

    return guide

def modify_language(guide, changes, model):
    """Apply specified changes to a constructed language."""

    click.echo(click.style(f"Modifying language using {model}...", dim=True))
    chat_completion = complete_chat(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "user", "content": f"Make the following changes to the constructed language outlined below. Your response should be an exhaustive reference sheet describing the new language. Assume the reader did not read the original reference sheet and that they have no prior experience using the language.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{changes}"}
        ]
    )
    improved_guide = chat_completion['choices'][0]['message']['content']

    return improved_guide

def reduce_dictionary(words, embeddings_model):
    """Remove similar words from a dictionary."""

    click.echo(click.style(f"Removing similar words using {embeddings_model}...", dim=True))

    # Retrieve the embeddings for each word
    word_embeddings = {word: _get_embeddings(word, embeddings_model) for word in words}

    # Remove similar words
    words_to_remove = set()
    for word_a, embedding_a in word_embeddings.items():
        for word_b, embedding_b in word_embeddings.items():
            if word_a != word_b and cosine_similarity(embedding_a, embedding_b) > 0.99:
                click.echo(click.style(f"Removed {word_b} because it is similar to {word_a}.", dim=True))
                words_to_remove.add(word_b)

    # Remove the similar words from the dictionary
    for word in words_to_remove:
        words.pop(word)

    return words

def create_dictionary(guide, mode, count, model, embeddings_model) -> dict:
    """Generate words for a constructed language."""

    # Choose a random topic
    if mode == "simple":
        # Generate words
        click.echo(click.style(f"Generating words using {model}...", dim=True))
        chat_completion = complete_chat(
            model=model,
            temperature=0.9,
            messages=[
                {"role": "user", "content": f"Generate {count} single random vocabulary words for the following constructed language. Format your response as a CSV document with two columns: Word and English Translation. Quote all cells in the CSV document.\n\nLanguage guide:\n\n{guide}"}
            ]
        )
        response = chat_completion['choices'][0]['message']['content']

    elif mode == "topic":
        click.echo(click.style(f"Choosing a random topic using {model}...", dim=True))
        chat_completion = complete_chat(
            model=model,
            temperature=0.9,
            messages=[
                {"role": "system", "content": "You are a writing assistant who virtually always suggests a different topic to write about."},
                {"role": "user", "content": f"Respond with a random topic to write about."},
            ]
        )
        topic = chat_completion['choices'][0]['message']['content']

        # Generate words
        click.echo(click.style(f"Generating words related to '{topic}' using {model}...", dim=True))
        chat_completion = complete_chat(
            model=model,
            temperature=0.9,
            messages=[
                {"role": "user", "content": f"Generate {count} single random vocabulary words related to '{topic}' for the following constructed language. Format your response as a CSV document with two columns: Word and English Translation. Quote all cells in the CSV document.\n\nLanguage guide:\n\n{guide}"}
            ]
        )
        response = chat_completion['choices'][0]['message']['content']

    else:
        raise Exception("Invalid mode. Please set the mode to 'simple' or 'topic'.")

    # Parse the generated words
    reader = csv.reader(response.splitlines())
    header = next(reader)
    if header != ["Word", "English Translation"]:
        raise Exception(f"Invalid response. Expected a CSV document with two rows: Word and Translation. Received: {response}")

    words = {}
    for row in reader:
        # Extract the word and translation
        if len(row) != 2:
            raise Exception(f"Invalid response. Expected row to have two columns: Word and Translation. Received: {row}")
        word, translation = row

        # If the word starts with a number (e.g., "1. hello"), remove the number
        if "." in word:
            first, rest = word.split(".", 1)
            if first.isdigit():
                word = rest.strip()

        # Add the word to the dictionary
        words[word] = translation

    if len(words) != count:
        click.echo(click.style(f"Warning: {len(words)} words were generated, but {count} were requested.", fg="yellow"))

    # Remove similar words
    words = reduce_dictionary(words, embeddings_model)

    return words

def merge_dictionaries(a, b, embeddings_model):
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

        if similarity > 0.99:
            if len(b_word) < len(a_word):
                click.echo(click.style(f"Removing {a_word} because it is too similar to {b_word}.", dim=True))
                del a[a_word]
            else:
                click.echo(click.style(f"Removing {b_word} because it is too similar to {a_word}.", dim=True))
                del b[b_word]

    # Merge the dictionaries
    merged = {**a, **b}

    return merged

import csv
import os

import click

from .language import generate_language, modify_language, improve_language, create_dictionary_for_text, merge_dictionaries, translate_text

@click.group()
def cli():
    pass

@cli.group()
def guide():
    pass

@guide.command()
@click.option("--design-goals", prompt="Enter the design goals of the language")
@click.option("--guide", "guide_path", prompt="Enter the filename to save the language guide to")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def create(design_goals, guide_path, model):
    """Create a constructed language."""

    # Generate language guide
    guide = generate_language(design_goals, model)

    # Save the language guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    click.echo(click.style(f"Language generated and saved to {guide_path} successfully.", dim=True))

@guide.command()
@click.option("--guide", "guide_path", prompt="Enter the filename of the language guide")
@click.option("--changes", prompt="Enter the changes to apply to the language guide")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def modify(guide_path, changes, model):
    """Make specific changes to the language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Update the language guide
    guide = modify_language(guide, changes, model)

    # Save the new guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    click.echo(click.style(f"Language modified and saved to {guide_path} successfully.", dim=True))

@guide.command()
@click.option("--guide", "guide_path", prompt="Enter the filename of the language guide")
@click.option("--dictionary", "dictionary_path", required=False, help="Enter the filename of the dictionary to use in 'example' mode.")
@click.option("--mode", default="simple", type=click.Choice(["simple", "example"]), help="Mode to use. Defaults to simple. Set to the experimental 'example' mode to include a new random translation in each revision.")
@click.option("-n", default=1, help="Number of revisions to perform. Defaults to 1.")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
@click.option("--embeddings-model", default="text-embedding-ada-002", help="OpenAI model to use for word embeddings in 'example' mode. Defaults to text-embedding-ada-002.")
def improve(guide_path, dictionary_path, mode, n, model, embeddings_model):
    """Automatically improve the language."""

    # Custom option validation
    if mode == "example":
        if dictionary_path is None:
            dictionary_path = click.prompt("Enter the filename of the dictionary")
    else:
        if dictionary_path is not None:
            raise click.BadParameter("The --dictionary option is not allowed in 'simple' mode.")

    if mode == "example":
        click.echo(click.style("Example mode is experimental. It may not work as expected.", fg="yellow"))

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Revise the language guide
    for i in range(n):
        guide = improve_language(guide, dictionary, model, embeddings_model, mode)

    # Save the improved guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    click.echo(click.style(f"Language improved and saved to {guide_path} successfully.", dim=True))

@cli.group()
def dictionary():
    pass

@dictionary.command()
@click.option("--guide", "guide_path", prompt="Enter the filename of the language guide")
@click.option("--dictionary", "dictionary_path", prompt="Enter the path to the CSV file to save the words to (will be created if it doesn't exist)")
@click.option("--text", prompt="Enter an English sentence to generate words for")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
@click.option("--embeddings-model", default="text-embedding-ada-002", help="OpenAI model to use for word embeddings. Defaults to text-embedding-ada-002.")
def add(guide_path, dictionary_path, text, model, embeddings_model):
    """Generate words in the language (experimental)."""

    click.echo(click.style("This feature is experimental. It may not work as expected.", fg="yellow"))

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load existing words
    if os.path.exists(dictionary_path):
        with open(output, "r") as file:
            reader = csv.reader(file)

            # Skip the header row
            next(reader)

            # Load existing words
            existing_words = {row[0]: row[1] for row in reader}
    else:
        existing_words = {}

    # Generate words
    words = create_dictionary_for_text(guide, text, existing_words, model, embeddings_model)
    click.echo(click.style(f"Generated {len(words)} words.", dim=True))

    # Combine the existing words with the new words, removing similar words
    click.echo(click.style("Removing similar words...", dim=True))
    all_words = merge_dictionaries(existing_words, words, embeddings_model)

    # Sort the words alphabetically
    all_words = {word: all_words[word] for word in sorted(all_words)}

    # Save the updated word list
    with open(dictionary_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Word", "Translation"])
        for word in all_words:
            writer.writerow([word, all_words[word]])

@cli.group()
def text():
    pass

@text.command()
@click.option("--guide", "guide_path", prompt="Enter the filename of the language guide")
@click.option("--dictionary", "dictionary_path", prompt="Enter the filename of the dictionary")
@click.option("--text", prompt="Enter the text to translate")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use")
@click.option("--embedding-model", default="text-embedding-ada-002", help="OpenAI model to use for word embeddings")
def translate(guide_path, dictionary_path, text, model, embedding_model):
    """Translate text to or from a constructed language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    with open(dictionary_path, "r") as file:
        reader = csv.reader(file)

        # Skip the header row
        next(reader)

        # Load the dictionary
        dictionary = {row[0]: row[1] for row in reader}

    # Translate the text
    translation = translate_text(text, guide, dictionary, model, embedding_model)
    click.echo(translation)

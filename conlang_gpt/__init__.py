import csv

import click

from .language import generate_language, modify_language, improve_language, create_dictionary_for_text, merge_dictionaries, load_dictionary, save_dictionary, translate_text

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
@click.option("--text", required=False)
@click.option("-n", default=1, help="Number of revisions to perform. Defaults to 1.")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
@click.option("--embeddings-model", default="text-embedding-ada-002", help="OpenAI model to use for word embeddings in 'example' mode. Defaults to text-embedding-ada-002.")
def improve(guide_path, dictionary_path, text, n, model, embeddings_model):
    """Automatically improve the language."""

    # Custom option validation
    if mode == "example":
        if dictionary_path is None:
            dictionary_path = click.prompt("Enter the filename of the dictionary")
    else:
        if dictionary_path is not None:
            raise click.BadParameter("The --dictionary option is not allowed in 'simple' mode.")

    if text:
        click.echo(click.style("The --text option is experimental. It may not work as expected.", fg="yellow"))

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Revise the language guide
    for i in range(n):
        # Try to improve the language guide
        improved_guide = improve_language(guide, dictionary, model, embeddings_model, text)

        # Stop if no problems were found
        if improved_guide is None:
            break

        # Update the language guide
        guide = improved_guide

    # Save the improved guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    click.echo(click.style(f"Language improved and saved to {guide_path} successfully.", dim=True))

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
    dictionary = load_dictionary(dictionary_path)

    while True:
        # Try to improve the language guide using the English text
        improved_guide = improve_language(guide, dictionary, model, embedding_model, text)

        # Stop if no problems were found
        if improved_guide is None:
            break

        # Update the language guide
        guide = improved_guide

    # Add any missing words to the dictionary
    related_words = create_dictionary_for_text(guide, text, dictionary, model, embedding_model)
    dictionary = merge_dictionaries(dictionary, related_words, embedding_model)

    # Translate the text
    translation = translate_text(text, guide, dictionary, model, embedding_model)
    click.echo(translation)

    # Save the updated guide
    with open(guide_path, "w") as file:
        file.write(guide)

    # Save the updated dictionary
    save_dictionary(dictionary, dictionary_path)

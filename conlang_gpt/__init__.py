import csv

import click

from .language import (
    ImproveDictionaryError,
    generate_language,
    modify_language,
    improve_language,
    create_dictionary_for_text,
    improve_dictionary,
    merge_dictionaries,
    load_dictionary,
    save_dictionary,
    translate_text,
)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--design-goals", prompt="Enter the design goals of the language")
@click.option(
    "--guide", "guide_path", prompt="Enter the filename to save the language guide to"
)
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="OpenAI model to use. Defaults to gpt-3.5-turbo.",
)
def create(design_goals, guide_path, model):
    """Create a constructed language."""

    # Generate language guide
    guide = generate_language(design_goals, model)

    # Save the language guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    click.echo(
        click.style(
            f"Language generated and saved to {guide_path} successfully.", dim=True
        )
    )


@cli.command()
@click.option(
    "--guide", "guide_path", prompt="Enter the filename of the language guide"
)
@click.option(
    "--dictionary",
    "dictionary_path",
    required=False,
    help="Enter the filename of the dictionary to use in 'example' mode.",
)
@click.option("--changes", prompt="Enter the changes to apply to the language guide")
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="OpenAI model to use. Defaults to gpt-3.5-turbo.",
)
@click.option(
    "--embeddings-model",
    default="text-embedding-ada-002",
    help="OpenAI model to use for word embeddings in 'example' mode. Defaults to text-embedding-ada-002.",
)
def modify(guide_path, dictionary_path, changes, model, embeddings_model):
    """Make specific changes to the language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    if dictionary_path is not None:
        dictionary = load_dictionary(dictionary_path)
    else:
        dictionary = {}

    # Update the language guide
    guide = modify_language(guide, changes, model)

    # Update the dictionary with the new guide
    dictionary = improve_dictionary(dictionary, guide, model, embeddings_model)

    # Save the new guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    click.echo(
        click.style(
            f"Language modified and saved to {guide_path} successfully.", dim=True
        )
    )

    # Save the new dictionary to a file
    if dictionary_path is not None:
        save_dictionary(dictionary, dictionary_path)

    click.echo(
        click.style(f"Dictionary saved to {dictionary_path} successfully.", dim=True)
    )


@cli.command()
@click.option(
    "--guide", "guide_path", prompt="Enter the filename of the language guide"
)
@click.option(
    "--dictionary",
    "dictionary_path",
    required=False,
    help="Enter the filename of the dictionary to use in 'example' mode.",
)
@click.option("--text", required=False)
@click.option(
    "--max-iterations",
    default=1,
    help="Max number of revisions to perform. Defaults to 1.",
)
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="OpenAI model to use. Defaults to gpt-3.5-turbo.",
)
@click.option(
    "--embeddings-model",
    default="text-embedding-ada-002",
    help="OpenAI model to use for word embeddings in 'example' mode. Defaults to text-embedding-ada-002.",
)
def improve(guide_path, dictionary_path, text, max_iterations, model, embeddings_model):
    """Automatically improve the language."""

    if text:
        click.echo(
            click.style(
                "The --text option is experimental. It may not work as expected.",
                fg="yellow",
            )
        )

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    dictionary = load_dictionary(dictionary_path)

    # Revise the language guide
    for i in range(max_iterations):
        # Try to improve the language guide
        improved_guide = improve_language(
            guide, dictionary, model, embeddings_model, text
        )

        # Stop if no problems were found
        if improved_guide is None:
            break

        # Update the language guide
        guide = improved_guide

        # Update the dictionary with the new guide
        while True:
            try:
                dictionary = improve_dictionary(
                    dictionary, guide, model, embeddings_model
                )
                break
            except ImproveDictionaryError as error:
                # If the dictionary can't be updated, it is most likely because
                # the guide is invalid. Try to fix the guide and then try
                # again.
                changes = f"The following problem(s) with the guide were encountered while updating the dictionary:\n\n{e}"
                click.echo(click.style(changes, fg="yellow"))
                modified_guide = modify_language(guide, changes, model)
                if modified_guide == guide:
                    click.echo(
                        click.style(
                            "The guide could not be modified to fix the problem(s).",
                            fg="yellow",
                        )
                    )
                else:
                    click.echo(
                        click.style(
                            "The guide was modified to fix the problem(s).", dim=True
                        )
                    )
                    guide = modified_guide

    # Save the improved guide to a file
    with open(guide_path, "w") as file:
        file.write(guide)

    # Save the new dictionary to a file
    save_dictionary(dictionary, dictionary_path)

    click.echo(
        click.style(
            f"Language improved and saved to {guide_path} successfully.", dim=True
        )
    )


@cli.command()
@click.option(
    "--guide", "guide_path", prompt="Enter the filename of the language guide"
)
@click.option(
    "--dictionary", "dictionary_path", prompt="Enter the filename of the dictionary"
)
@click.option("--text", prompt="Enter the text to translate")
@click.option("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
@click.option(
    "--embedding-model",
    default="text-embedding-ada-002",
    help="OpenAI model to use for word embeddings",
)
def translate(guide_path, dictionary_path, text, model, embedding_model):
    """Translate text to or from a constructed language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    dictionary = load_dictionary(dictionary_path)

    # Add any missing words to the dictionary
    related_words = create_dictionary_for_text(
        guide, text, dictionary, model, embedding_model
    )
    dictionary = merge_dictionaries(dictionary, related_words, embedding_model)

    while True:
        # Try to improve the language guide using the English text
        improved_guide = improve_language(
            guide, dictionary, model, embedding_model, text
        )

        # Stop if no problems were found
        if improved_guide is None:
            break

        # Update the language guide
        guide = improved_guide

        # Update the dictionary with the new guide
        while True:
            try:
                dictionary = improve_dictionary(
                    dictionary, guide, model, embedding_model
                )
                break
            except ImproveDictionaryError as e:
                # If the dictionary can't be updated, it is most likely because
                # the guide is invalid. Try to fix the guide and then try
                # again.
                changes = f"The following problem(s) with the guide were encountered while updating the dictionary:\n\n{e}"
                click.echo(click.style(changes, fg="yellow"))
                modified_guide = modify_language(guide, changes, model)
                if modified_guide == guide:
                    click.echo(
                        click.style(
                            "The guide could not be modified to fix the problem(s).",
                            fg="yellow",
                        )
                    )
                else:
                    click.echo(
                        click.style(
                            "The guide was modified to fix the problem(s).", dim=True
                        )
                    )
                    guide = modified_guide

    # Translate the text
    translation = translate_text(text, guide, dictionary, model, embedding_model)
    click.echo(translation)

    # Save the updated guide
    with open(guide_path, "w") as file:
        file.write(guide)

    # Save the updated dictionary
    save_dictionary(dictionary, dictionary_path)

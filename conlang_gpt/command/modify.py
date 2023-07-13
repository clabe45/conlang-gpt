import click

from ..language import (
    load_dictionary,
    improve_dictionary,
    modify_language,
    save_dictionary,
)


def modify(
    guide_path, dictionary_path, changes, similarity_threshold, model, embeddings_model
):
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
    dictionary = improve_dictionary(
        dictionary, guide, similarity_threshold, model, embeddings_model
    )

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

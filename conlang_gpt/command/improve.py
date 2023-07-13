import click

from ..language import (
    ImproveDictionaryError,
    improve_dictionary,
    improve_language,
    load_dictionary,
    modify_language,
)


def improve(
    guide_path,
    dictionary_path,
    max_iterations,
    similarity_threshold,
    model,
    embeddings_model,
):
    """Automatically improve the language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    dictionary = load_dictionary(dictionary_path)

    # Revise the language guide
    for i in range(max_iterations):
        # Try to improve the language guide
        improved_guide = improve_language(guide, dictionary, model, embeddings_model)

        # Stop if no problems were found
        if improved_guide is None:
            break

        # Update the language guide
        guide = improved_guide

        # Update the dictionary with the new guide
        while True:
            try:
                dictionary = improve_dictionary(
                    dictionary, guide, similarity_threshold, model, embeddings_model
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

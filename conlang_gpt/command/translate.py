import click

from ..embeddings import get_embeddings_model
from ..language import (
    ImproveDictionaryError,
    create_dictionary_for_text,
    generate_language,
    improve_dictionary,
    improve_language,
    load_dictionary,
    merge_dictionaries,
    modify_language,
    save_dictionary,
    translate_text,
)


def translate(
    guide_path,
    dictionary_path,
    text,
    max_improvements,
    similarity_threshold,
    model,
):
    """Translate text to or from a constructed language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    dictionary = load_dictionary(dictionary_path)

    # Create the embedding model
    embedding_model = get_embeddings_model()

    # Add any missing words to the dictionary
    new_words = create_dictionary_for_text(
        guide, text, dictionary, similarity_threshold, model, embedding_model
    )
    dictionary = merge_dictionaries(
        dictionary, new_words, similarity_threshold, embedding_model
    )

    for _ in range(max_improvements):
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
                    dictionary, guide, similarity_threshold, model, embedding_model
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
    _, explanation = translate_text(text, guide, dictionary, model, embedding_model)
    click.echo(explanation)

    # Save the updated guide
    with open(guide_path, "w") as file:
        file.write(guide)

    # Save the updated dictionary
    save_dictionary(dictionary, dictionary_path)

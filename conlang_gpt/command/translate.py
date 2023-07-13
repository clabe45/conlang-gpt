import click

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
    embedding_model,
):
    """Translate text to or from a constructed language."""

    # Load the beginner's guide
    with open(guide_path, "r") as file:
        guide = file.read()

    # Load the dictionary
    dictionary = load_dictionary(dictionary_path)

    # Add any missing words to the dictionary
    related_words = create_dictionary_for_text(
        guide, text, dictionary, similarity_threshold, model, embedding_model
    )
    dictionary = merge_dictionaries(
        dictionary, related_words, similarity_threshold, embedding_model
    )

    improvements_made = 0
    while improvements_made < max_improvements:
        # Try to improve the language guide using the English text
        improved_guide = improve_language(
            guide, dictionary, model, embedding_model, text
        )
        improvements_made += 1

        # Stop if no problems were found
        if improved_guide is None:
            break

        # Update the language guide
        guide = improved_guide

        # Update the dictionary with the new guide
        while improvements_made < max_improvements:
            try:
                dictionary = improve_dictionary(
                    dictionary, guide, similarity_threshold, model, embedding_model
                )
                improvements_made += 1
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

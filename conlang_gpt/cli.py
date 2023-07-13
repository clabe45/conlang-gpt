import csv

import click

from .command.create import create as create_
from .command.improve import improve as improve_
from .command.modify import modify as modify_
from .command.translate import translate as translate_


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

    create_(design_goals, guide_path, model)


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
    "--similarity-threshold",
    default=0.98,
    help="Maximum similarity between two words to be considered the same. Defaults to 0.98.",
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
def modify(
    guide_path, dictionary_path, changes, similarity_threshold, model, embeddings_model
):
    """Make specific changes to the language."""

    modify_(
        guide_path,
        dictionary_path,
        changes,
        similarity_threshold,
        model,
        embeddings_model,
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
@click.option(
    "--max-iterations",
    default=5,
    help="Max number of revisions to perform. Defaults to 5.",
)
@click.option(
    "--similarity-threshold",
    default=0.98,
    help="Maximum similarity between two words to be considered the same. Defaults to 0.98.",
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
def improve(
    guide_path,
    dictionary_path,
    max_iterations,
    similarity_threshold,
    model,
    embeddings_model,
):
    """Automatically improve the language."""

    improve_(
        guide_path,
        dictionary_path,
        max_iterations,
        similarity_threshold,
        model,
        embeddings_model,
    )


@cli.command()
@click.option(
    "--guide", "guide_path", prompt="Enter the filename of the language guide"
)
@click.option(
    "--dictionary", "dictionary_path", prompt="Enter the filename of the dictionary"
)
@click.option("--text", prompt="Enter the text to translate")
@click.option(
    "--max-improvements",
    default=5,
    help="Max number of relevant improvements to make to the guide and dictionary. Defaults to 5.",
)
@click.option(
    "--similarity-threshold",
    default=0.98,
    help="Maximum similarity between two words to be considered the same. Defaults to 0.98.",
)
@click.option("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
@click.option(
    "--embedding-model",
    default="text-embedding-ada-002",
    help="OpenAI model to use for word embeddings",
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

    translate_(
        guide_path,
        dictionary_path,
        text,
        max_improvements,
        similarity_threshold,
        model,
        embedding_model,
    )

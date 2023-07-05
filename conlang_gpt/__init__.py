import os

import click
import dotenv
import openai

from .language import generate_language, modify_language, improve_language, translate_text

dotenv.load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to your API key.")
openai.api_key = os.environ["OPENAI_API_KEY"]

@click.group()
def cli():
    pass

@cli.command()
@click.option("--design-goals", prompt="Enter the design goals of the language")
@click.option("--output-guide", prompt="Enter the filename to save the language guide to")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def create(design_goals, output_guide, model):
    """Create a constructed language."""

    # Generate language guide
    guide = generate_language(design_goals, model)

    # Save the language guide to a file
    with open(output_guide, "w") as file:
        file.write(guide)

    click.echo(f"Language generated and saved to {output_guide} successfully.")

@cli.command()
@click.option("--input-guide", prompt="Enter the filename of the language guide")
@click.option("--output-guide", prompt="Enter the filename to save the improved language guide to")
@click.option("--changes", prompt="Enter the changes to apply to the language guide")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def modify(input_guide, output_guide, changes, model):
    """Make specific changes to the language."""

    # Load the beginner's guide
    with open(input_guide, "r") as file:
        guide = file.read()

    # Update the language guide
    guide = modify_language(guide, changes, model)
    click.echo(f"Guide to updated language:\n\n{guide}\n")

    # Save the new guide to a file
    with open(output_guide, "w") as file:
        file.write(guide)

    click.echo(f"Language modified and saved to {output_guide} successfully.")

@cli.command()
@click.option("--input-guide", prompt="Enter the filename of the language guide")
@click.option("--output-guide", prompt="Enter the filename to save the improved language guide to")
@click.option("--mode", default="basic", type=click.Choice(["basic", "example"]), help="Mode to use. Defaults to basic. Set to the experimental 'example' mode to include a new random translation in each revision.")
@click.option("--steps", default=3, help="Number of revisions to perform. Defaults to 3.")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def improve(input_guide, output_guide, mode, steps, model):
    """Automatically improve the language."""

    # Load the beginner's guide
    with open(input_guide, "r") as file:
        guide = file.read()

    # Revise the language guide
    for i in range(steps):
        guide = improve_language(guide, model, mode)

    # Save the improved guide to a file
    with open(output_guide, "w") as file:
        file.write(guide)

    click.echo(f"Language improved and saved to {output_guide} successfully.")

@cli.command()
@click.option("--guide", prompt="Enter the filename of the language guide")
@click.option("--text", prompt="Enter the text to translate")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use")
def translate(guide, text, model):
    """Translate text into a constructed language."""

    # Load the beginner's guide
    with open(guide, "r") as file:
        guide = file.read()

    # Translate the text
    translation = translate_text(text, guide, model)
    click.echo(translation)

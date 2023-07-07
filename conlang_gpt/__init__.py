import csv
import os

import click

from .language import generate_language, modify_language, improve_language, generate_words, translate_text

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

    click.echo(click.style(f"Language generated and saved to {output_guide} successfully.", dim=True))

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

    # Save the new guide to a file
    with open(output_guide, "w") as file:
        file.write(guide)

    click.echo(click.style(f"Language modified and saved to {output_guide} successfully.", dim=True))

@cli.command()
@click.option("--input-guide", prompt="Enter the filename of the language guide")
@click.option("--output-guide", prompt="Enter the filename to save the improved language guide to")
@click.option("--mode", default="basic", type=click.Choice(["basic", "example"]), help="Mode to use. Defaults to basic. Set to the experimental 'example' mode to include a new random translation in each revision.")
@click.option("-n", default=1, help="Number of revisions to perform. Defaults to 1.")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def improve(input_guide, output_guide, mode, n, model):
    """Automatically improve the language."""

    if mode == "example":
        click.echo(click.style("Example mode is experimental. It may not work as expected.", fg="yellow"))

    # Load the beginner's guide
    with open(input_guide, "r") as file:
        guide = file.read()

    # Revise the language guide
    for i in range(n):
        guide = improve_language(guide, model, mode)

    # Save the improved guide to a file
    with open(output_guide, "w") as file:
        file.write(guide)

    click.echo(click.style(f"Language improved and saved to {output_guide} successfully.", dim=True))

@cli.command()
@click.option("--guide", prompt="Enter the filename of the language guide")
@click.option("--output", prompt="Enter the path to the CSV file to save the words to (will be created if it doesn't exist)")
@click.option("-n", default=15, help="Number of words to generate. Defaults to 15.")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use. Defaults to gpt-3.5-turbo-16k.")
def words(guide, output, n, model):
    """Generate words in the language (experimental)."""

    click.echo(click.style("This feature is experimental. It may not work as expected.", fg="yellow"))

    # Load the beginner's guide
    with open(guide, "r") as file:
        guide = file.read()

    # Generate words
    words = generate_words(guide, model, n)

    # Save the words to a CSV file, appending to the file if it already exists
    # Load existing words
    if os.path.exists(output):
        with open(output, "r") as file:
            reader = csv.reader(file)

            # Skip the header row
            next(reader)

            # Load existing words
            existing_words = {row[0]: row[1] for row in reader}
    else:
        existing_words = {}

    # Combine the existing words with the new words, removing duplicates
    all_words = existing_words | words

    # Sort the words alphabetically
    all_words = {word: all_words[word] for word in sorted(all_words)}

    # Save the updated word list
    with open(output, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Word", "Translation"])
        for word in all_words:
            writer.writerow([word, all_words[word]])

    click.echo(click.style(f"{len(words)} words generated and saved to {output} successfully.", dim=True))

@cli.command()
@click.option("--guide", prompt="Enter the filename of the language guide")
@click.option("--text", prompt="Enter the text to translate")
@click.option("--model", default="gpt-3.5-turbo-16k", help="OpenAI model to use")
def translate(guide, text, model):
    """Translate text to or from a constructed language."""

    # Load the beginner's guide
    with open(guide, "r") as file:
        guide = file.read()

    # Translate the text
    translation = translate_text(text, guide, model)
    click.echo(translation)

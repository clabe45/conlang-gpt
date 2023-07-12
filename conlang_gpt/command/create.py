import click

from ..language import generate_language


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

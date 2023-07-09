import os
import time

import click
import dotenv
import openai


dotenv.load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise Exception(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to your API key."
    )
openai.api_key = os.environ["OPENAI_API_KEY"]


def complete_chat(**kwargs):
    """Complete a chat with the OpenAI API."""

    delay = 1
    while True:
        try:
            completion = openai.ChatCompletion.create(**kwargs)
            break
        except openai.error.APIError as e:
            click.echo(
                click.style(
                    f"OpenAI API error: {e}. Retrying in {delay} second(s)...", fg="red"
                )
            )
            time.sleep(delay)
        except openai.error.RateLimitError:
            click.echo(
                click.style(
                    f"OpenAI API rate limit exceeded. Retrying in {delay} second(s)...",
                    fg="red",
                )
            )
            time.sleep(delay)
            delay *= 2
        except openai.error.ServiceUnavailableError:
            click.echo(
                click.style(
                    f"OpenAI API service unavailable. Retrying in {delay} second(s)...",
                    fg="red",
                )
            )
            time.sleep(delay)
        except openai.error.Timeout:
            click.echo(
                click.style(
                    f"OpenAI API timeout. Retrying in {delay} second(s)...", fg="red"
                )
            )
            time.sleep(delay)

    return completion


def get_embedding(text, model):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

import os

import dotenv
import openai


dotenv.load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to your API key.")
openai.api_key = os.environ["OPENAI_API_KEY"]


def complete_chat(**kwargs):
    """Complete a chat with the OpenAI API."""

    return openai.ChatCompletion.create(**kwargs)


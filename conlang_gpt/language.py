import click
import openai


def translate_text(text, language_guide, model):
    """Translate text into a constructed language."""

    click.echo(click.style(f"Translating text using {model}...", dim=True))
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": f"Translate the text below from or into the following constructed language. Show your work in detail.\n\nLanguage guide:\n\n{language_guide}\n\nText to translate:\n\n{text}"}]
    )

    translation = chat_completion['choices'][0]['message']['content']
    return translation

def generate_english_text(model):
    chat_completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        messages=[
            {"role": "system", "content": "You are an unpredictable writing assistant."},
            {"role": "user", "content": "Please generate a random English sentence."}
        ]
    )

    english_text = chat_completion['choices'][0]['message']['content']
    return english_text

def improve_language(guide, model, mode):
    click.echo(click.style(f"Improving language using {model}...", dim=True))

    if mode == "basic":
        # Identify problems with the language
        chat_completion = openai.ChatCompletion.create(
            model=model,
            temperature=0.5,
            presence_penalty=0.5,
            messages=[{"role": "user", "content": f"Please identify one flaw, contradiction or point of confusion with the language outlined below along with detailed, actionable steps to fix it.\n\nLanguage guide:\n\n{guide}"}]
        )
        revisions = chat_completion['choices'][0]['message']['content']

    elif mode == "example":
        # Attempt to translate a random English sentence
        english_text = generate_english_text(model)
        translation = translate_text(english_text, guide, model)
        click.echo(f"Sample English text:\n\n{english_text}\n")
        click.echo(f"Translated text:\n\n{translation}\n")

        # Identify problems with the language using the translated text as an example/reference
        chat_completion = openai.ChatCompletion.create(
            model=model,
            temperature=0.1,
            messages=[{"role": "user", "content": f"Please identify one flaw or point of confusion with the language outlined below along with detailed, actionable steps to fix it. I included a sample translation to give you more context.\n\nLanguage guide:\n\n{guide}\n\nSample English text: {english_text}\n\nTranslated text: {translation}"}]
        )
        revisions = chat_completion['choices'][0]['message']['content']
    else:
        raise Exception("Invalid mode. Please set the mode to 'basic' or 'example'.")

    click.echo(f"Change:\n\n{revisions}\n")

    # Rewrite the language guide
    chat_completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "user", "content": f"Improve the following constructed language to address the problem described below. Your response should be an exhaustive reference sheet describing the new language. Assume the reader did not read the original reference sheet and that they have no prior experience using the language.\n\nOriginal language guide:\n\n{guide}\n\nMake this change:\n\n{revisions}"}
        ]
    )
    improved_guide = chat_completion['choices'][0]['message']['content']
    click.echo(f"Guide to improved language:\n\n{guide}\n")

    return improved_guide

def generate_language(design_goals, model):
    """Generate a constructed language."""

    click.echo(f"Generating language using {model}...")
    chat_completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "user", "content": f"Create a constructed language with the following design goals:\n\n{design_goals}\n\nYour response should be an exhaustive reference sheet including all the information that is needed to use the language. Assume the reader has no prior experience using the language."}
        ],
    )
    guide = chat_completion['choices'][0]['message']['content']
    click.echo(f"Initial draft:\n\n{guide}\n")

    return guide

def modify_language(guide, changes, model):
    """Apply specified changes to a constructed language."""

    click.echo(click.style(f"Modifying language using {model}...", dim=True))
    chat_completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "user", "content": f"Make the following changes to the constructed language outlined below. Your response should be an exhaustive reference sheet describing the new language. Assume the reader did not read the original reference sheet and that they have no prior experience using the language.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{changes}"}
        ]
    )
    improved_guide = chat_completion['choices'][0]['message']['content']
    click.echo(f"Guide to improved language:\n\n{guide}\n")

    return improved_guide

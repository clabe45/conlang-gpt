import csv

from conlang_gpt.language import create_dictionary_for_text


def test_create_dictionary_for_text_adds_all_required_words_to_empty_dictionary(guide):
    dictionary = create_dictionary_for_text(
        guide, "Hello", {}, 0.98, "gpt-4", "text-embedding-ada-002"
    )

    assert len(dictionary) == 1


def test_create_dictionary_for_text_returns_dictionary_with_all_missing_words_when_some_words_already_have_translations(
    guide,
):
    dictionary = create_dictionary_for_text(
        guide,
        "Hello, world.",
        {"E": "Hello"},
        0.98,
        "gpt-4",
        "text-embedding-ada-002",
    )

    assert len(dictionary) == 1
    assert "world" in [word.lower() for word in dictionary.values()]


def test_create_dictionary_for_text_returns_empty_dictionary_when_all_words_are_already_translated(
    guide,
):
    dictionary = create_dictionary_for_text(
        guide,
        "Hello, world.",
        {"E": "Hello", "I": "world"},
        0.98,
        "gpt-4",
        "text-embedding-ada-002",
    )

    assert len(dictionary) == 0

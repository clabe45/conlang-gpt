import csv

from openai.embeddings_utils import cosine_similarity
import pytest

from conlang_gpt.language import (
    create_dictionary_for_text,
    improve_dictionary,
    merge_dictionaries,
    reduce_dictionary,
    translate_text,
)
from conlang_gpt.openai import get_embedding


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


def test_create_dictionary_for_text_does_not_generate_names_of_people(guide):
    dictionary = create_dictionary_for_text(
        guide,
        "Bob walks",
        {"u": "walks"},
        0.98,
        "gpt-4",
        "text-embedding-ada-002",
    )

    assert len(dictionary) == 0


def test_improve_dictionary_updates_words_that_do_not_follow_guide_rules(
    guide,
):
    improved_dictionary = improve_dictionary(
        {"C": "Hello"},
        guide,
        0.98,
        "gpt-4",
        "text-embedding-ada-002",
    )

    assert (
        len(improved_dictionary) == 1
        and set(improved_dictionary.values()) == {"Hello"}
        and set(improved_dictionary.keys()) != {"C"}
    )


@pytest.mark.parametrize(
    "word1, translation1, word2, translation2",
    [
        ("E", "Hello", "I", "Hello"),
        ("E", "Hello", "I", "Hi"),
        ("E", "Be", "I", "Am"),
        ("E", "Be", "I", "Is"),
        ("E", "Be", "I", "Are"),
    ],
)
def test_reduce_dictionary_removes_similar_words(
    word1, translation1, word2, translation2
):
    dictionary = reduce_dictionary(
        {word1: translation1, word2: translation2}, 0.9, "text-embedding-ada-002"
    )

    assert len(dictionary) == 1


@pytest.mark.parametrize(
    "word1, translation1, word2, translation2",
    [
        ("E", "Hello", "I", "Goodbye"),
    ],
)
def test_reduce_dictionary_does_not_remove_distinct_words(
    word1, translation1, word2, translation2
):
    dictionary = reduce_dictionary(
        {word1: translation1, word2: translation2}, 0.9, "text-embedding-ada-002"
    )

    assert len(dictionary) == 2


@pytest.mark.parametrize(
    "word1, translation1, word2, translation2",
    [
        ("E", "Hello", "I", "Hello"),
        ("E", "Hello", "I", "Hi"),
        ("E", "Be", "I", "Am"),
        ("E", "Be", "I", "Is"),
        ("E", "Be", "I", "Are"),
    ],
)
def test_merge_dictionaries_merges_words_with_similar_translations(
    word1, translation1, word2, translation2
):
    dictionary = merge_dictionaries(
        {word1: translation1}, {word2: translation2}, 0.9, "text-embedding-ada-002"
    )

    assert len(dictionary) == 1


@pytest.mark.parametrize(
    "word1, translation1, word2, translation2",
    [
        ("E", "Hello", "I", "Goodbye"),
    ],
)
def test_merge_dictionaries_does_not_merge_words_with_distinct_translations(
    word1, translation1, word2, translation2
):
    dictionary = merge_dictionaries(
        {word1: translation1}, {word2: translation2}, 0.9, "text-embedding-ada-002"
    )

    assert len(dictionary) == 2


def test_translate_text_translated_translation_is_similar_to_original_text(
    guide,
):
    original_text = "Hello, world."
    translated_text = original_text
    for _ in range(2):
        translated_text, _ = translate_text(
            translated_text,
            guide,
            {"e": "Hello", "o^": "world"},
            "gpt-4",
            "text-embedding-ada-002",
        )

    # Compute cosine similarity between original and translated text
    original_embedding = get_embedding(original_text, "text-embedding-ada-002")
    translated_embedding = get_embedding(translated_text, "text-embedding-ada-002")
    similarity = cosine_similarity(original_embedding, translated_embedding)
    assert similarity > 0.9

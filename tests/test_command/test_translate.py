import csv

from conlang_gpt.command.translate import translate


def test_translate_with_improvements_disabled_populates_empty_dictionary(
    guide_path, dictionary_path
):
    translate(
        guide_path,
        dictionary_path,
        "hi",
        0,
        0.98,
        "gpt-4",
        "text-embedding-ada-002",
    )

    csv_reader = csv.reader(dictionary_path.open("r"))
    # Skip the header row
    next(csv_reader)
    assert len(list(csv_reader)) > 0

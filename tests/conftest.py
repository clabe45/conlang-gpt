import csv

import pytest


DEFAULT_GUIDE = """# Language Name: Pentalit

Letters: A, E, I, O, U

Design Principles:

1. The Language:
Pentalit is a vowel-only, tonal language. The meaning of syllables changes depending on the intonations used.

2. Vowel Combinations:
Pentalit uses 5 letters, which can be combined in sequences of up to 3 letters. If repetition is allowed, this results in 125 unique combinations. These combinations can be single vowel, two vowels, or three vowels; e.g., A, EA, IAI.

3. Tones:
Pentalit uses four tones: high, low, rising, and falling. These tones significantly increase the number of potential meanings for each combination of letters.

4. Word Formation:
Words in Pentalit may consist of one to four syllables, corresponding roughly to monosyllabic, disyllabic, trisyllabic, and quadrisyllabic words.

Reference Sheet:

1. Pronunciation Guide and Tone Markers:
High: Marked with an acute accent (´), pronounced with high pitch.
Low: No marker, pronounced with the lowest pitch.
Rising: Marked with a caron (^), starts low and rises to a high pitch.
Falling: Marked with a grave accent (`), starts high and falls down to low.

2. Basic Grammar:
Pentalit uses Subject-Verb-Object (SVO) word order. The language does not use grammatical genders, articles, or plurals, simplifying its complexity.

3. Syntax Rules:
a) Affirmative Sentence: 
Subject + Verb + Object 
Example: A E^I O
Translation: I eat fruit

b) Negative Sentence:
Subject + Verb + negation + Object
Example: A E^I U O
Translation: I do not eat fruit

c) Interrogative Sentence:
"AI" + Statement
Example: AI A E^I O
Translation: Do I eat fruit?

Remember, fluency and proficiency in Pentalit will take time and practice, as with any language. Despite its reduced set of phonemes, the usage of tones and combination of vowels gives Pentalit enough flexibility to be a functional constructed language."""


@pytest.fixture()
def guide():
    return DEFAULT_GUIDE


@pytest.fixture()
def guide_path(tmp_path, guide):
    path = tmp_path / "guide.md"
    path.write_text(guide)
    return path


@pytest.fixture()
def dictionary():
    return {}


@pytest.fixture()
def dictionary_path(tmp_path, dictionary):
    path = tmp_path / "dictionary.csv"
    csv_writer = csv.writer(path.open("w"))
    csv_writer.writerow(["Word", "Translation"])
    for word, translation in dictionary.items():
        csv_writer.writerow([word, translation])
    return path

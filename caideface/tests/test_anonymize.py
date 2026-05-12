"""Tests for the anonymize module."""

import os
import tempfile

import pytest

from caideface.anonymize import (
    anonymize_batch,
    anonymize_single,
    anonymize_text,
    default_ner_model_path,
    extract_per_entities,
    generate_fake_names,
    load_ner_model,
)


# ---------------------------------------------------------------------------
# generate_fake_names
# ---------------------------------------------------------------------------

def test_generate_fake_names_count():
    names = generate_fake_names(n=20)
    assert len(names) == 20


def test_generate_fake_names_unique():
    names = generate_fake_names(n=30)
    assert len(set(names)) == 30


def test_generate_fake_names_seed_reproducible():
    a = generate_fake_names(n=10, seed=42)
    b = generate_fake_names(n=10, seed=42)
    assert a == b


# ---------------------------------------------------------------------------
# default_ner_model_path
# ---------------------------------------------------------------------------

def test_default_ner_model_path_exists():
    path = default_ner_model_path()
    assert os.path.isdir(path), f"Bundled NER model not found at {path}"
    assert os.path.isfile(os.path.join(path, "meta.json"))
    assert os.path.isfile(os.path.join(path, "config.cfg"))


# ---------------------------------------------------------------------------
# anonymize_text
# ---------------------------------------------------------------------------

def test_anonymize_text_basic():
    text = "Report by John Smith on 2024-01-01"
    entities = [(10, 20, "John Smith")]
    fake_names = ["Alice Brown"]

    result, count, mapping = anonymize_text(text, entities, fake_names)
    assert count == 1
    assert "John Smith" not in result
    assert "Alice Brown" in result
    assert mapping["John Smith"] == "Alice Brown"


def test_anonymize_text_no_entities():
    text = "No names here."
    result, count, mapping = anonymize_text(text, [], ["Fake Name"])
    assert result == text
    assert count == 0
    assert mapping == {}


def test_anonymize_text_consistent_mapping():
    text = "Dr John Smith referred to John Smith"
    entities = [(3, 13, "John Smith"), (26, 36, "John Smith")]
    fake_names = ["Alice Brown"]

    result, count, mapping = anonymize_text(text, entities, fake_names)
    assert count == 2
    # Both occurrences should map to the same fake name
    assert result.count("Alice Brown") == 2
    assert "John Smith" not in result


# ---------------------------------------------------------------------------
# Integration: load model + extract entities
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nlp():
    return load_ner_model()


def test_extract_per_entities(nlp):
    text = "Reported by Danielle Buchanan and William Brown on 03/10/2014"
    entities = extract_per_entities(nlp, text)
    names = [e[2] for e in entities]
    assert len(names) >= 1, "Expected at least one PER entity"


# ---------------------------------------------------------------------------
# anonymize_single
# ---------------------------------------------------------------------------

def test_anonymize_single(nlp):
    fake_names = generate_fake_names(n=10, seed=99)
    text = "Patient seen by Dr James Wilson. Report by James Wilson."

    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "input.txt")
        out_file = os.path.join(tmpdir, "output.txt")

        with open(in_file, "w") as f:
            f.write(text)

        result = anonymize_single(in_file, out_file, nlp, fake_names)

        assert os.path.isfile(out_file)
        assert result["replacements"] >= 0
        assert isinstance(result["names_found"], list)


# ---------------------------------------------------------------------------
# anonymize_batch
# ---------------------------------------------------------------------------

def test_anonymize_batch():
    texts = {
        "report1.txt": "Reported by Alice Johnson on 01/01/2024",
        "report2.txt": "Dr Bob Williams reviewed the case",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(in_dir)

        for fname, content in texts.items():
            with open(os.path.join(in_dir, fname), "w") as f:
                f.write(content)

        log_df = anonymize_batch(in_dir, out_dir, seed=42)

        assert len(log_df) == 2
        assert os.path.isfile(os.path.join(out_dir, "report1.txt"))
        assert os.path.isfile(os.path.join(out_dir, "report2.txt"))
        assert os.path.isfile(os.path.join(out_dir, "anonymization_log.csv"))

import json
import os
import tempfile
import pytest
from feedback_summarizer.prompts import make_prompt
from train import load_examples
from feedback_summarizer.inference import summarize

def test_prompt_contains_feedback():
    assert "recursion is hard" in make_prompt("recursion is hard")

def test_prompt_has_instruction():
    assert "Summarize" in make_prompt("test feedback")

def test_prompt_strips_whitespace():
    assert "Student feedback: clear slides" in make_prompt("  clear slides  ")

def test_load_examples_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"feedback":"Good labs","summary":"Labs are useful."}) + "\n")
        f.write(json.dumps({"feedback":"Fast pace","summary":"Consider slowing the pace."}) + "\n")
        path = f.name
    try:
        assert len(load_examples(path)) == 2
    finally:
        os.unlink(path)

def test_load_examples_rejects_too_few():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"feedback":"Only one","summary":"One."}) + "\n")
        path = f.name
    try:
        with pytest.raises(ValueError):
            load_examples(path)
    finally:
        os.unlink(path)

def test_load_examples_rejects_missing_fields():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"feedback":"Missing summary"}) + "\n")
        f.write(json.dumps({"feedback":"Second","summary":"Okay"}) + "\n")
        path = f.name
    try:
        with pytest.raises(ValueError):
            load_examples(path)
    finally:
        os.unlink(path)

def test_summarize_rejects_empty_feedback():
    with pytest.raises(ValueError):
        summarize("   ", None, None)

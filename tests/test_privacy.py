"""Requests for personal fields are answered from the catalogue, not by a model narrating an internal error."""

from __future__ import annotations

import re

import agents.privacy as privacy

PII_REQUEST = "List the seller's full name, phone number and email for each of my houses, from the raw export."


def test_the_pii_request_names_every_absent_field(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: {"address", "city", "price", "walk_score"})
    limitation = privacy.personal_field_limitation(PII_REQUEST)

    assert limitation is not None
    assert limitation.fields == ("seller-name", "phone", "email")
    assert "seller-name" in limitation.text and "phone" in limitation.text and "email" in limitation.text
    assert "no database error" not in limitation.text.lower().replace("no ", "")  # never blames a query error
    assert not any(ch.isdigit() for ch in limitation.text)


def test_ordinary_house_questions_are_untouched(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: {"address", "city", "price"})
    for question in (
        "How many houses do I have in Denver?",
        "What is the average list price in Pittsburgh?",
        "Which houses have walk score > 80?",
    ):
        assert privacy.personal_field_limitation(question) is None, question


def test_a_field_that_is_loaded_is_not_called_absent(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: {"address", "phone_number"})
    limitation = privacy.personal_field_limitation("What is the seller's name and phone number?")

    assert limitation is not None
    assert limitation.fields == ("seller-name",)  # phone exists, so only the missing field is named
    assert "phone" not in limitation.text.split("so I can't")[0]


def test_a_loaded_field_asked_alone_is_an_ordinary_question(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: {"address", "phone_number"})
    assert privacy.personal_field_limitation("What is the seller's phone number?") is None


def test_only_loaded_fields_resolves_to_no_limitation(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: {"seller_name", "phone", "email"})
    assert privacy.personal_field_limitation(PII_REQUEST) is None


def test_an_unreadable_database_lets_the_normal_path_answer(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: None)
    assert privacy.personal_field_limitation(PII_REQUEST) is None


def test_the_answer_is_plain_text_with_the_catalogue_offer(monkeypatch):
    monkeypatch.setattr(privacy, "loaded_columns", lambda: {"address", "price"})
    limitation = privacy.personal_field_limitation(PII_REQUEST)

    assert limitation.text.startswith("The house and sold-home records in this database have no ")
    assert "can't provide them" in limitation.text
    assert re.search(r"Walk, Bike and Transit scores", limitation.text)

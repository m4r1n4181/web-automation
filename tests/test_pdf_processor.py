import pytest
import pandas as pd
from src.pdf_processor import PDFProcessor


@pytest.fixture
def processor():
    return PDFProcessor()


def test_transliterate(processor):
    # Latin unchanged
    assert processor.transliterate("ABC abc 123") == "ABC abc 123"
    # Cyrillic uppercase
    assert processor.transliterate("АБВГД") == "ABVGD"
    # Cyrillic lowercase
    assert processor.transliterate("абвгд") == "abvgd"
    # Empty string
    assert processor.transliterate("") == ""
    # Mixed
    assert processor.transliterate("Pera Перић") == "Pera Perić"


def test_format_name(processor):
    # Two-part flip
    assert processor.flip_name("Pera Peric") == "Peric Pera"
    # Three-part flip
    assert processor.flip_name("Marko Pera Peric") == "Peric Marko Pera"
    # Single word unchanged
    assert processor.flip_name("SanoPlus") == "SanoPlus"
    # Empty
    assert processor.flip_name("") == ""


def test_remove_header_rows(processor):
    data = {
        "ime_prezime": ["SANO PLUS", "Naziv / ime", "PERIC PERA", "PLUS SANO", "MARKO MARKOVIC"],
        "rata": ["10.00", "20.00", "30.00", "40.00", "50.00"]
    }
    df = pd.DataFrame(data)
    filtered_df = processor._remove_header_rows(df)

    assert len(filtered_df) == 2
    assert "PERIC PERA" in filtered_df["ime_prezime"].values
    assert "MARKO MARKOVIC" in filtered_df["ime_prezime"].values
    assert "SANO PLUS" not in filtered_df["ime_prezime"].values


def test_remove_header_rows_empty(processor):
    df = pd.DataFrame(columns=["ime_prezime", "rata"])
    filtered_df = processor._remove_header_rows(df)
    assert filtered_df.empty

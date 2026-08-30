from retrieval.bm25 import normalize_text, tokenize


def test_persian_character_normalization():
    assert normalize_text("كتاب يک") == "کتاب یک"
    assert tokenize("سلام، دنیا!") == ["سلام", "دنیا"]

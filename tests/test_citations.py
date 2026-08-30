from rag.citations import audit_citations


def test_citation_audit_accepts_valid_and_flags_invalid():
    audit = audit_citations("Fact [S1], another [S3], bad [S9].", source_count=3)
    assert audit.cited_indices == (1, 3, 9)
    assert audit.invalid_indices == (9,)
    assert audit.has_valid_citation is True


def test_citation_audit_handles_no_citations():
    audit = audit_citations("No citation here", source_count=2)
    assert audit.has_valid_citation is False
    assert audit.invalid_indices == ()

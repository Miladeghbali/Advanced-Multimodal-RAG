from __future__ import annotations

import re
from dataclasses import dataclass


_SOURCE_RE = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True)
class CitationAudit:
    cited_indices: tuple[int, ...]
    invalid_indices: tuple[int, ...]
    has_valid_citation: bool


def audit_citations(answer: str, source_count: int) -> CitationAudit:
    cited = tuple(sorted({int(x) for x in _SOURCE_RE.findall(answer)}))
    invalid = tuple(index for index in cited if index < 1 or index > source_count)
    valid = tuple(index for index in cited if 1 <= index <= source_count)
    return CitationAudit(
        cited_indices=cited,
        invalid_indices=invalid,
        has_valid_citation=bool(valid),
    )

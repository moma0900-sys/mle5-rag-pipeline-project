import re

from qdrant_client import models


NO_ANSWER = "Found results are not good enough to provide an answer."


def validate_citations(
    answer: str,
    evidence: list[models.ScoredPoint],
) -> None:
    # Welche Quellen wurden tatsächlich abgerufen?
    allowed_sources = {
        hit.payload["doc_id"]
        for hit in evidence
    }

    # Welche Quellen stehen in eckigen Klammern in der Antwort?
    cited_sources = set(
        re.findall(r"\[([^\[\]\r\n]+)\]", answer)
    )

    if not cited_sources:
        raise ValueError("Answer has no sources.")

    # Mengen-Differenz: zitierte, aber nicht erlaubte Quellen
    unknown_sources = cited_sources - allowed_sources

    if unknown_sources:
        raise ValueError(
            f"Not used sources: {sorted(unknown_sources)}"
        )



# Verarbeitet die bekannten Qdrant-Treffer
def generate_answer(
    evidence: list[models.ScoredPoint],
) -> str:
    if not evidence:
        return NO_ANSWER

    passages = []

    for hit in evidence:
        payload = hit.payload

        passages.append(
            f"{payload['text']} [{payload['doc_id']}]"
        )

    answer = "\n\n".join(passages)

    validate_citations(answer, evidence)

    return answer
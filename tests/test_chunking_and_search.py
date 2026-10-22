'''Tests für das Chunking und die Qdrant-Suche'''
from pathlib import Path

import numpy as np
import pytest

from helpers import Chunk, Document
from read_md_files import chunk_documents
from vector_store import COLLECTION_NAME, store_chunks



# Der Text wird mit einem Wort Überlappung aufgeteil
# Dokument-ID und Titel bleiben erhalten
# Die Chunk-Position zählt innerhalb eines Dokuments hoch und beginnt beim nächsten wieder bei 0
# --> Test stellt sicher, dass beim Aufteilen weder Text noch die Zuordnung zur Quelle verloren geht
def test_chunk_metadata_is_preserved():
    documents = [
        Document(
            doc_id="refunds",
            title="Refunds",
            text="eins zwei drei vier fuenf sechs\n\nWeiterer Absatz.",
            path=Path("refunds.md"),
        ),
        Document(
            doc_id="account",
            title="Account",
            text="Anderer Inhalt.",
            path=Path("account.md"),
        ),
    ]

    chunks = chunk_documents(
        documents,
        max_words=4,
        overlapping_words=1,
    )

    assert [
        (c.doc_id, c.title, c.position, c.chunk_id)
        for c in chunks
    ] == [
        ("refunds", "Refunds", 0, "refunds-0"),
        ("refunds", "Refunds", 1, "refunds-1"),
        ("refunds", "Refunds", 2, "refunds-2"),
        ("account", "Account", 0, "account-0"),
    ]

    assert [c.text for c in chunks] == [
        "eins zwei drei vier",
        "vier fuenf sechs",
        "Weiterer Absatz.",
        "Anderer Inhalt.",
    ]



# Speichert zwei künstliche Vektoren: refunds-0 = [1, 0], account-0 = [0, 1]
# --> Die Suchanfrage [0, 1] muss den Account-Chunk liefern, einschließlich seines vollständigen Payloads
# Test zeigt, dass Speichern und Suchen technisch korrekt zusammenspielen
# Qdrant findet den passenden Vektor und liefert den dazugehörigen Text samt Quelleninformationen zurück
# Vorgegeben: Refunds → [1, 0], Account → [0, 1]
def test_qdrant_returns_matching_vector_and_metadata():
    chunks = [
        Chunk(
            chunk_id="refunds-0",
            doc_id="refunds",
            title="Refunds",
            text="Refund details.",
            position=0,
        ),
        Chunk(
            chunk_id="account-0",
            doc_id="account",
            title="Account",
            text="Account details.",
            position=0,
        ),
    ]

    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    client = store_chunks(chunks, vectors)

    try:
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            # suche nach identischem Vektor [0, 1] --> also suche nach Account[0, 1]
            query=[0.0, 1.0],
            limit=1,
            with_payload=True,
            with_vectors=False,
        ).points

        assert len(hits) == 1
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].payload == chunks[1].model_dump(mode="json")

    finally:
        client.close()
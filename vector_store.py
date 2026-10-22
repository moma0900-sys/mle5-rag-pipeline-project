import numpy as np
from qdrant_client import QdrantClient, models

from helpers import Chunk


COLLECTION_NAME = "northstar_support"


def store_chunks(
    chunks: list[Chunk],
    vectors: np.ndarray,
) -> QdrantClient:

    # Anzahl und Form der Vektoren kontrollieren
    if (
        not chunks
        or vectors.ndim != 2
        or vectors.shape[0] != len(chunks)
        or vectors.shape[1] < 1
    ):
        raise ValueError(
            "Each Chunk needs exactly one Vector."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Found invalid figures in vectors."
        )

    points = []

    # Je einen Chunk und seinen Vektor zusammenführen und mit fortlaufender Nummer versehen
    # zip() verbindet Elemente an selber Position von Liste vector[] und Liste chunk[] --> (vector[0], chunk[0]), (vector[1], chunk[1]), ...
    # enumerate macht daraus (0, (vector[0], chunk[0])), (1, (vector[1], chunk[1])), ...
    for point_id, (chunk, vector) in enumerate(
        zip(chunks, vectors, strict=True)
    ):
        # .tolist() wandelt einen NumPy-Vektor in eine normale Python-Liste um
        # e.g. vector = np.array([0.2, -0.4, 0.8], dtype=np.float32) ---> [0.20000000298023224, -0.4000000059604645, 0.800000011920929]
        point = models.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            # wandelt ein Chunk-Objekt in ein Dictionary um und übergibt es als Zusatzdaten an Qdrant
            payload=chunk.model_dump(mode="json")
        )
        points.append(point)

    # Lokale Datenbank im Arbeitsspeicher starten
    client = QdrantClient(":memory:")

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                # size gibt an, dass genau zwei Zahlen in vectors[] stehen müssen: [Chunks, Vektoren], zb. [22, 384]
                size=vectors.shape[1],
                distance=models.Distance.COSINE,
            ),
        )

        # .upsert = UPdate and inSERT --> schreibt points[] in die Qdrant-Collection
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
    except Exception:
        client.close()
        raise

    # Rückgabewert ist fertige Qdrant-DB, die im Arbeitsspeicher unter "northstar_support" abgelegt wird
    return client



# Keine Treffer?                 → leere Liste
# Bester Score unter 0,45?       → leere Liste
# Sonst: Passagen ab 0,40 wählen → Liste mit Kontextkandidaten
def select_evidence(
    hits: list[models.ScoredPoint],
    min_top_score: float = 0.45,
    min_chunk_score: float = 0.40,
) -> list[models.ScoredPoint]:

    if not 0 <= min_top_score <= 1:
        raise ValueError(
            f"Best answer score result has to be betwenn 0 and  1 --> actual value: {min_top_score}"
    )

    if not 0 <= min_chunk_score <= 1:
        raise ValueError(
            f"Best chunk score result has to be betwenn 0 and  1 --> actual value: {min_chunk_score}"
        )

    if min_chunk_score > min_top_score:
        raise ValueError(
            "Threshold for chunk score has to be lower than answer score --> Change definition values in vector_store.py"
            f"Chunk-Threshold: {min_chunk_score}; "
            f"Score of best result: {min_top_score}"
        )

    # Keine Treffer vorhanden
    if not hits:
        return []

    # Höchsten Suchwert bestimmen
    # .score kommt aus Qdrant Bibliothek
    best_score = max(hit.score for hit in hits)

    # Insgesamt keine ausreichende Suchbasis
    if best_score < min_top_score:
        return []

    # Einzelne Kontextpassagen auswählen
    # hits limitiert durch  limit = 3
    return [
        hit
        for hit in hits
        if hit.score > 0 and hit.score >= min_chunk_score
    ]
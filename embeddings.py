import numpy as np
from sentence_transformers import SentenceTransformer

from helpers import Chunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )

# Hugging Face → Modelldateien im lokalen Cache (HDD)
#                             ↓
#                      Modell im RAM
#                             ↓
#                  .encode(texts) auf der CPU
#                             ↓
#                     Vektoren im RAM


# Von Wörtern zu Zahlenvektoren
def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    if not chunks:
        raise ValueError("No chunks for embedding.")

    # Title und Text in Liste texts:
    # texts[0] = "Password Reset Links" \n "Each reset link expires after 20 minutes."
    texts = [
        f"{chunk.title}\n{chunk.text}"
        for chunk in chunks
    ]

    # Für jedes Element in texts einen Vektor berechnen
    # model = MiniLM, .encode dessen Methode zur Berechnung der Embeddings --> Texte in Vektoren umwandlen
    # normalize_embeddings = Vektoren auf Länge 1 normieren
    # Normalization shows symmetrie better: A = [3, 4], B = [6, 8] --> A = [0.6, 0.8], B = [0.6, 0.8]
    # Advantage: 1: Große Zahlenwerte allein sollen einen Treffer nicht bevorzugen, 2: Vereinfacht die Rechnung
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    # np-Array mit Form (22, 384) --> 22 Chunks, pro Chunk 384 Vektoren
    return np.asarray(vectors, dtype=np.float32)


'''Ein Chunk: Titel + Text
          ↓
In Tokens zerlegen + Token-Vektoren im Transformer verarbeiten
          ↓
Token-Repräsentationen zusammenfassen („Pooling“)
          ↓
Ein Vektor für den gesamten Chunk
          ↓
Normieren und zurückgeben: 384 Zahlen'''




# macht das selbe wie funktion embed_chunks --> Tokenize/vektorize --> Normalize
def embed_question(
    question: str,
    model: SentenceTransformer,
) -> np.ndarray:

    if not question.strip():
        raise ValueError("No empty question allowed.")

    vector = model.encode(
        question.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(vector, dtype=np.float32)
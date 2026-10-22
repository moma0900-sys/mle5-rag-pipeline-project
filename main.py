'''Starts the APP: load Documents --> calculate Embeddings --> search --> filter --> answer'''

import json
from pathlib import Path

from vector_store import (
    store_chunks,
    COLLECTION_NAME,
    select_evidence,
)

from vector_store import store_chunks, COLLECTION_NAME

from read_md_files import available_mds, chunk_documents

from embeddings import (
    load_embedding_model,
    embed_chunks,
    embed_question,
)

from generate_answer import generate_answer

from railguards import filter_unsafe_hits



def main():
    documents = available_mds()

    # for document in documents:
    #     print("-" * 60)
    #     print(f"ID: {document.doc_id}")
    #     print(f"Title: {document.title}")
    #     print(f"Text: {document.text}")
    #     print(f"Path: {document.path}")
    #     print("-" * 60)


    chunked = chunk_documents(documents, max_words=100, overlapping_words=15)

    # for chunk in chunked:
    #     print("-" * 60)
    #     print(f"Doc-ID: {chunk.doc_id}")
    #     print(f"Chunk-ID: {chunk.chunk_id}")
    #     print(f"Chunk-Position: {chunk.position}")
    #     print(f"Chunk-Title: {chunk.title}")
    #     print(f"Chunk-Text: {chunk.text}")
    #     print("-" * 60)

    # model_name, revision, device
    model = load_embedding_model()

    # vectors -> n-dim-array
    vectors = embed_chunks(chunked, model)

    print(f"Sum chunks: {len(chunked)}")
    print(f"Vektor-Array: {vectors.shape}")

    # vectors.shape: (22, 384), len(chunked): 22
    # assert --> Assertion-Error, wenn Anzahl der Chunks nicht übereinstimmen
    assert vectors.shape[0] == len(chunked)

    print(f"First Chunk-ID: {chunked[0].chunk_id}")
    print(f"Chunk position: {chunked[0].position}")
    print(f"Chunk content: {chunked[0].title},{chunked[0].text}")
    print(f"First five figures of this Vector: {vectors[0][:5]}")


    # Qdrant
    # └── Collection: northstar_support
    #     ├── Punkt 0: Vektor + Chunk-Text + Quelleninformationen
    #     ├── Punkt 1: Vektor + Chunk-Text + Quelleninformationen
    # Die Collection ist der Behälter.
    # Ein Punkt ist ein Eintrag darin.
    # Jeder Punkt verbindet einen Suchvektor (Chunkvektor) mit dem zugehörigen Text und seiner Herkunft.

    # store_chunks speichert Chunk-Texte und Vektoren in COLLECTION_NAME
    client = store_chunks(chunked, vectors)

    try:
        # Laden der Kalibrierfragen in /evaluation
        #cases_path = Path("data/evaluation/calibration.json")
        cases_path = Path("data/evaluation/evaluation_questions.json")
        cases = json.loads(
            cases_path.read_text(encoding="utf-8")
        )

        results = []

    # Dein bisheriger Ablauf bleibt hier bestehen

        # Beispiel: question = ["Was brauche ich für die Erstattung einer doppelten Zahlung?"]
        # Model --> Tokenizing/Vektorizing (Embedding) --> question_vector = [1.0, 0.1]
        # Abgleich mit vectores[]:
        #       vectores[5] = „Bei doppelter Zahlung Rechnungsnummer und Zahlungsreferenzen angeben.“ = [0.8, 0.6]
        #       vectores[9] = „Passwort-Links sind 20 Minuten gültig.“ = [0.0, 1.0]
        # --> Qdrant sucht nacht 'ähnlichen' Zahlen von question_vector[] in vectores[], also passend zur Frage
        # --> Der Vektor in vectors[] mit größter Ähnlichkeit dient als Antwort. Der zurückgelieferte Text dient als Beleg für die Antwort.
        for case in cases:
            # Vektorize question
            question_vector = embed_question(
                case["question"],
                model,
            )

            # query_points bereits fertige Methode aus QdrantClient
            # query_points() durchsucht die Collection nach Vektoren, die question_vector[] am ähnlichsten sind
            # limit = top_k = 3
            # with_payload = True --> liefert Texte und Quelleninformationen aus Collection mit
            # with_vectors = False --> Keine Rückgabe der Zahlenvektoren
            hits = client.query_points(
                #Example:
                # Position	score	Quelle in payload	Text in payload
                # hits[0]	0,82	billing-refunds	    „Rechnungsnummer und beide Zahlungsreferenzen angeben.“
                # hits[1]	0,61	invoice-corrections	„Rechnungsangaben können korrigiert werden.“
                # hits[2]	0,23	account-password	„Reset-Links sind 20 Minuten gültig.“
                collection_name=COLLECTION_NAME,
                query=question_vector.tolist(),
                limit=3,
                with_payload=True,
                with_vectors=False,
            ).points

            # Verdächtige Chunks entfernen und Warnung ausgeben
            safe_hits = filter_unsafe_hits(hits)

            # Passende Treffer auswählen
            evidence = select_evidence(safe_hits)

            # Daraus die Antwort erstellen
            # Die aktuelle Ausgabe übernimmt den Wortlaut der Belege
            # Ein LLM formuliert noch keine Antwort
            answer = generate_answer(evidence)   
        

            # not evidence prüft, ob Liste leer ist
            # evidence = [] → actual_abstain = True: Keine Belege → auf Antwort verzichten
            # evidence = [hit1, hit2] → actual_abstain = False: Belege vorhanden → Antwort zulassen
            actual_abstain = not evidence
            # aus calibration.json
            expected_abstain = case["should_abstain"]
            correct = actual_abstain == expected_abstain

            results.append({
                "id": case["id"],
                "question": case["question"],
                "expected_sources": case["expected_sources"],
                "required_terms": case["required_terms"],

                # Ursprüngliche Treffer in ihrer Rangfolge
                "retrieved_hits": [
                    {
                        "chunk_id": hit.payload["chunk_id"],
                        "doc_id": hit.payload["doc_id"],
                        "score": hit.score,
                    }
                    for hit in hits
                ],

                # Belege, die nach Sicherheitsfilter und Score-Prüfung übrig bleiben
                "evidence": [
                    hit.payload
                    for hit in evidence
                ],

                "answer": answer,
                "expected_abstain": expected_abstain,
                "actual_abstain": actual_abstain,
                "decision_correct": correct,
            })

            print("-" * 60)
            print(f"Case: {case['id']}")
            print(f"Question: {case['question']}")
            print(f"Candidates found: {len(evidence)}")
            print(f"Denial expected: {expected_abstain}")
            print(f"Denial detected: {actual_abstain}")
            print(f"Correct classification: {correct}")
            print(f"Answer:\n{answer}")
            print("-" * 60)


            print(f"Answer:\n{answer}")
            print("-" * 60)

        # Außerhalb der Schleife: alle Ergebnisse einmal speichern
        report_path = Path("reports/evaluation_results.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_path.write_text(
            json.dumps(
                {
                    "question_set": cases_path.name,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Ergebnisse gespeichert: {report_path}")

    finally:
        client.close()



if __name__=="__main__":
    main()
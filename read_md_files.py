'''Check available MD-files'''
from pathlib import Path

from helpers import Document, Chunk, ChunkSize


def available_mds() -> list[Document]:
    md_dir = Path("data/knowledge_base")

    documents: list[Document] = []

    for md in sorted(md_dir.iterdir()):
        if md.is_file() and md.suffix.lower() == ".md":

            #strip: Remove leading or trailing whitespaces, e.g.: "   Hello.   " --> "Hello."
            raw_text = md.read_text(encoding="utf-8").strip()
            lines = raw_text.splitlines()

            if not lines or not lines[0].startswith("# "):
                raise ValueError(f"No Markdown title in {md}")
            
            title = lines[0].removeprefix("# ").strip()
            body = "\n".join(lines[1:]).strip()

            if not title or not body:
                raise ValueError(f"Empty Title or Body in {md}")

            # md.name    --> "billing-refunds.md"
            # md.stem    --> "billing-refunds"
            # md.suffix  --> ".md"
            documents.append(Document(doc_id=md.stem, title=title, text=body, path=md))

    if not documents:
        raise ValueError(f"No MD files found in {md_dir}")

    return documents



def chunk_size(chunks: ChunkSize) -> list[str]:

    if (chunks.max_words < 1 or not 0 < chunks.overlapping_words < chunks.max_words):
        raise ValueError(
        "Rules:"
        "max_words > 0; "
        "0 < overlapping_words < max_words"
    )
    
    words = chunks.text.split()

    # e.g. words = 50, max_words = 100
    if len(words) <= chunks.max_words:
        return [chunks.text]

    step = chunks.max_words - chunks.overlapping_words

    chunked = []

    # e.g. words = 150, max_words = 100, overlapping_words = 10 --> step = 90
    # range('start', 'stop', 'step')
    # range(0, (150-10, 1), 90) --> range (0, (140, 1), 90)  --> wenn 'stop' < 1 ist, dann 1
    for start in range(0, max(len(words) - chunks.overlapping_words, 1), step):
        passage = words[start : start + chunks.max_words]

        if passage: #wenn nicht leer, dann
            chunked.append(" ".join(passage)) #join words[0] to words[99]

    return chunked



def chunk_documents(documents: list[Document], max_words: int = 100, overlapping_words: int = 12) -> list[Chunk]:
    # both max_words and overlap_words needs to be int
    if (
        not isinstance(max_words, int)
        or not isinstance(overlapping_words, int)
        or max_words < 1
        or not 0 < overlapping_words < max_words
    ):
        raise ValueError(
            "Rules:"
            "max_words > 0; "
            "0 < overlapping_words < max_words"
        )
    
    chunks: list[Chunk] = []

    for document in documents:
        paragraphs = [
            passage.replace("\n", " ").strip()
            for passage in document.text.split("\n\n")
            if passage.strip()
        ]

        position = 0

        for paragraph in paragraphs:
            config = ChunkSize(
                text=paragraph,
                max_words=max_words,
                overlapping_words=overlapping_words,
            )

            for chunk in chunk_size(config):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-{position}",
                        doc_id=document.doc_id,
                        title=document.title,
                        text=chunk,
                        position=position
                    )
                )
                position += 1

    return chunks
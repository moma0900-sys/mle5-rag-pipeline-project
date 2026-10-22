from pathlib import Path

from pydantic import BaseModel


class Document(BaseModel):
    '''MD files format after readout'''
    doc_id: str
    title: str
    text: str
    path: Path


class Chunk(BaseModel):
    '''Necessary Chunk format'''
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int


class ChunkSize(BaseModel):
    text: str
    max_words: int
    overlapping_words: int
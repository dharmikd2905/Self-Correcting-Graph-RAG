"""
Phase 1 -- Ingestion & Knowledge Extraction Pipeline (PRD 4.1).

1. Splits raw text with RecursiveCharacterTextSplitter (800 / 150, per PRD).
2. Extracts normalized (Subject, Predicate, Object) triples per chunk using
   an LLM constrained to the `TripleExtractionResult` Pydantic schema.
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings, get_settings
from app.schemas.models import Chunk, Triple, TripleExtractionResult

logger = logging.getLogger(__name__)

TRIPLE_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge-graph extraction engine.
Read the text chunk and extract factual Knowledge Triples of the form
(Subject, Predicate, Object) that capture relationships between entities,
components, systems, or concepts mentioned in the text.

Rules:
- Normalize entity names (consistent casing/naming across triples).
- Predicates should be short, snake_case verb phrases (e.g. "depends_on", "causes", "part_of").
- Only extract triples explicitly supported by the text. Do not infer facts not present.
- If no clear relational facts exist in the chunk, return an empty list.
- Extract at most 8 triples per chunk.
"""


def chunk_document(doc_id: str, source_name: str, text: str, settings: Settings | None = None) -> list[Chunk]:
    """Splits document text into overlapping chunks."""
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    chunks: list[Chunk] = []
    for i, raw in enumerate(raw_chunks):
        chunk_id = f"{doc_id}_{i}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"
        chunks.append(Chunk(chunk_id=chunk_id, text=raw, source=source_name, doc_id=doc_id))
    return chunks


def extract_triples_for_chunk(chunk: Chunk, llm: BaseChatModel) -> list[Triple]:
    """Extract triples from a single chunk using structured LLM output."""
    structured_llm = llm.with_structured_output(TripleExtractionResult)
    try:
        result: TripleExtractionResult = structured_llm.invoke(
            [
                ("system", TRIPLE_EXTRACTION_SYSTEM_PROMPT),
                ("human", f"Text chunk:\n\n{chunk.text}"),
            ]
        )
        return result.triples
    except Exception as exc:  # noqa: BLE001
        logger.warning("Triple extraction failed for chunk %s: %s", chunk.chunk_id, exc)
        return []


def extract_triples_for_document(chunks: list[Chunk], llm: BaseChatModel, max_workers: int = 8) -> list[Chunk]:
    """
    Populates chunks with `.triples` in parallel using a ThreadPoolExecutor
    to drastically speed up multi-chunk document/PDF processing.
    """
    if not chunks:
        return chunks

    def _process_chunk(chunk: Chunk) -> Chunk:
        chunk.triples = extract_triples_for_chunk(chunk, llm)
        return chunk

    workers = min(max_workers, len(chunks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_process_chunk, chunks))

    return chunks
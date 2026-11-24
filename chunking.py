from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from typing import Sequence

import numpy as np
import requests
from pypdf import PdfReader


OLLAMA = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:4b"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
INDEX_PATH = DOCS_DIR / "ips_embeddings.npz"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 32
INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    page: int

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}"


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = max(start + 1, end - overlap)
        while start < len(text) and text[start].isspace():
            start += 1

    return chunks


def extract_chunks(pdf_files: Sequence[Path]) -> list[TextChunk]:
    chunks = []
    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = re.sub(r"\s+", " ", page_text).strip()
            chunks.extend(
                TextChunk(text=text, source=pdf_file.name, page=page_number)
                for text in split_text(page_text)
            )

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDFs. "
            "Scanned PDFs may need OCR before they can be indexed."
        )
    return chunks


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot create embeddings for an empty list.")

    batches = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start:start + EMBED_BATCH_SIZE])
        response = requests.post(
            f"{OLLAMA}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise ValueError("Ollama returned an unexpected embedding response.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Ollama returned invalid embedding vectors.")
        batches.append(vectors)

    return np.concatenate(batches, axis=0)


def _fingerprint(pdf_files: Sequence[Path]) -> str:
    digest = sha256()
    settings = {
        "format_version": INDEX_FORMAT_VERSION,
        "embedding_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for pdf_file in pdf_files:
        digest.update(b"\0")
        digest.update(pdf_file.name.encode("utf-8"))
        with pdf_file.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_index(chunks: Sequence[TextChunk], vectors: np.ndarray) -> None:
    if not chunks or vectors.ndim != 2 or len(chunks) != len(vectors):
        raise ValueError("The saved document chunks and embeddings do not match.")
    if not np.isfinite(vectors).all():
        raise ValueError("The saved index contains invalid embedding vectors.")


def load_or_build_index(
    docs_dir: Path = DOCS_DIR,
    index_path: Path = INDEX_PATH,
) -> tuple[list[TextChunk], np.ndarray, bool]:
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF documents found in {docs_dir}.")

    fingerprint = _fingerprint(pdf_files)
    expected_metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "embedding_model": EMBED_MODEL,
        "fingerprint": fingerprint,
    }

    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as saved_index:
            metadata = json.loads(saved_index["metadata"].item())
            if metadata == expected_metadata:
                chunks = [
                    TextChunk(**item)
                    for item in json.loads(saved_index["chunks"].item())
                ]
                vectors = np.asarray(saved_index["vectors"], dtype=np.float32)
                _validate_index(chunks, vectors)
                return chunks, vectors, False

    chunks = extract_chunks(pdf_files)
    vectors = embed_texts([chunk.text for chunk in chunks])
    _validate_index(chunks, vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(expected_metadata, sort_keys=True)
    chunks_json = json.dumps([asdict(chunk) for chunk in chunks])
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=index_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                metadata=np.asarray(metadata_json),
                chunks=np.asarray(chunks_json),
                vectors=vectors,
            )
        temporary_path.replace(index_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return chunks, vectors, True


if __name__ == "__main__":
    indexed_chunks, _, created = load_or_build_index()
    action = "Created" if created else "Loaded"
    print(
        f"{action} an index of {len(indexed_chunks)} chunks "
        f"from {len({chunk.source for chunk in indexed_chunks})} PDF document(s) "
        f"at {INDEX_PATH}."
    )

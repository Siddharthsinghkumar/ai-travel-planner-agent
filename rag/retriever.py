"""Minimal RAG retriever using sentence-transformers + numpy (no Chroma)."""

import os
import glob
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional


class RAGRetriever:
    """Embeds a corpus directory and retrieves top-k chunks by cosine similarity."""

    def __init__(self, corpus_dir: str, collection_name: str = "travel_kb"):
        from sentence_transformers import SentenceTransformer

        self.corpus_dir = corpus_dir
        self.collection_name = collection_name
        self.cache_dir = os.path.join(".rag_embeddings", collection_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load_or_build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Return top_k chunks as [{"text": ..., "source": ..., "score": ...}]."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (self.embeddings @ q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": round(s, 4),
            })
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_or_build(self):
        """Load from cache if valid, otherwise rebuild."""
        cache_path = os.path.join(self.cache_dir, "embeddings.npy")
        meta_path = os.path.join(self.cache_dir, "metadata.json")
        hash_path = os.path.join(self.cache_dir, "corpus_hash.txt")

        current_hash = self._corpus_hash()
        if os.path.exists(hash_path):
            with open(hash_path) as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash and os.path.exists(cache_path) and os.path.exists(meta_path):
                self.embeddings = np.load(cache_path)
                with open(meta_path) as f:
                    self.chunks = json.load(f)
                return

        self._build()
        np.save(cache_path, self.embeddings)
        with open(meta_path, "w") as f:
            json.dump(self.chunks, f)
        with open(hash_path, "w") as f:
            f.write(current_hash)

    def _build(self):
        """Read, chunk, embed all files in corpus_dir."""
        files = self._collect_files()
        raw_chunks: list[dict] = []
        for fpath in files:
            text = self._read_file(fpath)
            if not text:
                continue
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                raw_chunks.append({"text": chunk, "source": os.path.basename(fpath), "chunk_index": i})

        self.chunks = raw_chunks
        if raw_chunks:
            texts = [c["text"] for c in raw_chunks]
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        else:
            self.embeddings = np.array([])

    def _collect_files(self) -> list[str]:
        """Find all .txt, .md, .pdf files in corpus_dir."""
        patterns = ["*.txt", "*.md", "*.pdf"]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(self.corpus_dir, "**", pat), recursive=True))
        return sorted(files)

    def _read_file(self, fpath: str) -> str:
        """Read text from a file. PDF extraction tries pypdf, falls back to empty."""
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(fpath)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def _read_pdf(fpath: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(fpath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return ""

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks of ~chunk_size tokens."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start = end - overlap
        return chunks

    def _corpus_hash(self) -> str:
        """Hash all files in corpus_dir to detect changes."""
        h = hashlib.sha256()
        for fpath in sorted(self._collect_files()):
            h.update(fpath.encode())
            try:
                with open(fpath, "rb") as f:
                    h.update(f.read())
            except Exception:
                pass
        return h.hexdigest()

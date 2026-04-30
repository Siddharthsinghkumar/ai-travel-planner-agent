import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class RAGRetriever:
    def __init__(self, corpus_dir: str = "rag/corpus"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks, self.sources = [], []
        for f in Path(corpus_dir).glob("**/*.md"):
            text = f.read_text()
            for i in range(0, len(text), 1500):
                chunk = text[i:i+1800].strip()
                if chunk:
                    self.chunks.append(chunk)
                    self.sources.append(f.name)
        if self.chunks:
            self.embeddings = self.model.encode(self.chunks, normalize_embeddings=True)
        else:
            self.embeddings = np.zeros((0, 384))

    def retrieve(self, query: str, top_k: int = 4):
        if len(self.chunks) == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(scores)[-top_k:][::-1]
        return [{"text": self.chunks[i], "source": self.sources[i], "score": float(scores[i])} for i in top]

"""
Experiment 2: RAG-based Question Answering System
====================================================
Implements the three classic RAG stages as separate, inspectable components:

  1. INDEXING   -> DocumentIndexer  (load, chunk, embed, store)
  2. RETRIEVAL  -> Retriever        (embed query, similarity search, top-k)
  3. GENERATION -> Generator        (build prompt from context, call LLM)

Works with or without extra libraries installed:
  - Embeddings: sentence-transformers if available, else TF-IDF (sklearn) fallback.
  - Vector store: FAISS if available, else a plain NumPy cosine-similarity index.
  - Generation: OpenAI API if OPENAI_API_KEY is set, else Anthropic if
    ANTHROPIC_API_KEY is set, else a simple extractive fallback (no key needed,
    so you can test indexing+retrieval end-to-end without any API key).

Author: generated for coursework use.
"""

import os
import re
import glob
import argparse
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 0. Data structures
# ----------------------------------------------------------------------

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source: str = ""


# ----------------------------------------------------------------------
# 1. Embedding backend (auto-selects best available)
# ----------------------------------------------------------------------

class Embedder:
    """Wraps sentence-transformers (preferred) or TF-IDF (fallback)."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.mode = "tfidf"
            self._fitted = False

    def fit(self, texts: List[str]):
        """Only needed for TF-IDF mode; must be called once on the corpus."""
        if self.mode == "tfidf":
            self.vectorizer.fit(texts)
            self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.mode == "sentence-transformers":
            return self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
        else:
            if not self._fitted:
                # Fit-on-the-fly fallback (not ideal, but keeps single-query calls working)
                self.fit(texts)
            vecs = self.vectorizer.transform(texts).toarray()
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (vecs / norms).astype("float32")


# ----------------------------------------------------------------------
# 2. INDEXING
# ----------------------------------------------------------------------

class DocumentIndexer:
    """
    Loads raw documents, splits them into overlapping chunks, embeds each
    chunk, and stores everything in a vector index (FAISS if available,
    else a NumPy matrix used for brute-force cosine similarity).
    """

    def __init__(self, embedder: Embedder, chunk_size: int = 500, chunk_overlap: int = 100):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = None
        self._faiss_index = None
        self._use_faiss = False

    # ---- loading ----
    @staticmethod
    def load_documents(folder: str) -> Dict[str, str]:
        """Reads every .txt/.md file in `folder` into {filename: text}."""
        docs = {}
        for path in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
            if path.lower().endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    docs[os.path.basename(path)] = f.read()
        return docs

    # ---- chunking ----
    def _chunk_text(self, text: str) -> List[str]:
        """Simple sliding-window chunker over whitespace-split tokens."""
        words = re.split(r"\s+", text.strip())
        words = [w for w in words if w]
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(words), step):
            piece = words[start:start + self.chunk_size]
            if not piece:
                break
            chunks.append(" ".join(piece))
            if start + self.chunk_size >= len(words):
                break
        return chunks or [text.strip()]

    # ---- build index ----
    def build(self, docs: Dict[str, str]):
        self.chunks = []
        for doc_id, text in docs.items():
            for i, ch_text in enumerate(self._chunk_text(text)):
                self.chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=ch_text, source=doc_id))

        if not self.chunks:
            raise ValueError("No chunks produced — check that the docs folder has .txt/.md files.")

        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)  # no-op for sentence-transformers
        self.embeddings = self.embedder.encode(texts)

        try:
            import faiss
            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
            index.add(self.embeddings)
            self._faiss_index = index
            self._use_faiss = True
        except Exception:
            self._use_faiss = False  # numpy brute-force fallback

        print(f"[Indexing] {len(self.chunks)} chunks from {len(docs)} docs "
              f"| embedder={self.embedder.mode} | vector_store={'faiss' if self._use_faiss else 'numpy'}")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]


# ----------------------------------------------------------------------
# 3. RETRIEVAL
# ----------------------------------------------------------------------

class Retriever:
    def __init__(self, indexer: DocumentIndexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        q_vec = self.indexer.embedder.encode([query])

        if self.indexer._use_faiss:
            scores, idxs = self.indexer._faiss_index.search(q_vec, top_k)
            results = [(self.indexer.chunks[i], float(s))
                       for i, s in zip(idxs[0], scores[0]) if i != -1]
        else:
            sims = (self.indexer.embeddings @ q_vec[0])  # cosine sim (vectors are normalized)
            top_idx = np.argsort(-sims)[:top_k]
            results = [(self.indexer.chunks[i], float(sims[i])) for i in top_idx]

        return results


# ----------------------------------------------------------------------
# 4. GENERATION
# ----------------------------------------------------------------------

class Generator:
    """
    Builds a grounded prompt from retrieved chunks and calls an LLM.
    Falls back to a template-based extractive answer if no API key is set,
    so the pipeline is always runnable end-to-end.
    """

    SYSTEM_PROMPT = (
        "You are a precise question-answering assistant. Answer the user's "
        "question using ONLY the provided context. If the answer is not in "
        "the context, say you don't have enough information. Cite the source "
        "filename(s) you used in square brackets."
    )

    def __init__(self):
        self.backend = "extractive-fallback"
        if os.environ.get("OPENAI_API_KEY"):
            self.backend = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.backend = "anthropic"

    def _build_prompt(self, query: str, contexts: List[Tuple[Chunk, float]]) -> str:
        ctx_block = "\n\n".join(
            f"[{c.source} #chunk{c.chunk_id}] {c.text}" for c, _ in contexts
        )
        return f"Context:\n{ctx_block}\n\nQuestion: {query}\n\nAnswer:"

    def generate(self, query: str, contexts: List[Tuple[Chunk, float]]) -> str:
        prompt = self._build_prompt(query, contexts)

        if self.backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        if self.backend == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        # ---- extractive fallback: no API key needed ----
        best_chunk, best_score = contexts[0]
        sources = ", ".join(sorted({c.source for c, _ in contexts}))
        return (
            f"(No LLM API key found — showing extractive fallback answer.)\n"
            f"Most relevant passage (score={best_score:.3f}):\n"
            f"\"{best_chunk.text[:600]}\"\n"
            f"[sources: {sources}]"
        )


# ----------------------------------------------------------------------
# 5. Full pipeline
# ----------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, top_k: int = 4):
        self.embedder = Embedder()
        self.indexer = DocumentIndexer(self.embedder, chunk_size, chunk_overlap)
        self.retriever = None
        self.generator = Generator()
        self.top_k = top_k

    def index(self, docs_folder: str):
        docs = DocumentIndexer.load_documents(docs_folder)
        if not docs:
            raise FileNotFoundError(f"No .txt/.md files found in {docs_folder}")
        self.indexer.build(docs)
        self.retriever = Retriever(self.indexer)

    def ask(self, query: str) -> Dict:
        if self.retriever is None:
            raise RuntimeError("Call .index(docs_folder) before .ask(query)")
        contexts = self.retriever.retrieve(query, top_k=self.top_k)
        answer = self.generator.generate(query, contexts)
        return {
            "query": query,
            "answer": answer,
            "retrieved": [
                {"source": c.source, "chunk_id": c.chunk_id, "score": s, "text": c.text}
                for c, s in contexts
            ],
        }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 2: RAG-based QA system")
    parser.add_argument("--docs", type=str, default="./docs", help="Folder of .txt/.md documents")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    args = parser.parse_args()

    pipeline = RAGPipeline(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, top_k=args.top_k)
    pipeline.index(args.docs)
    result = pipeline.ask(args.query)

    print("\n=== Retrieved chunks ===")
    for r in result["retrieved"]:
        print(f"- [{r['source']} #{r['chunk_id']}] score={r['score']:.3f}\n  {r['text'][:150]}...")

    print("\n=== Answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()

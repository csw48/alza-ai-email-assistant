import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# (texts, task_type) -> vektory; v produkcii GeminiService.embed
Embedder = Callable[[List[str], str], List[List[float]]]

DOCS_DIR = Path(__file__).parent / "documents"


def _tokens(text: str) -> set:
    """Malé písmená, bez diakritiky a interpunkcie, skrátené na prvé 4 znaky.
    Hrubý stemming pre slovenské/české skloňovanie: 'dronom' ~ 'drone', 'doručenie' ~ 'doručovanie'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return {w[:4] for w in re.findall(r"\w+", text) if len(w) > 2}


def split_queries(text: str, max_parts: int = 8) -> List[str]:
    """Celý text + jednotlivé vety/otázky (bez krátkych fragmentov ako pozdrav či podpis)."""
    parts = [p.strip() for p in re.split(r"(?<=[?!.])\s+|\n+", text) if len(p.strip()) >= 15]
    return [text] + parts[:max_parts]


class KnowledgeBase:
    """
    Malý RAG nad markdown dokumentmi Alzy.
    Dokumenty sa delia podľa '## ' sekcií. Ak je dostupný embedder (Gemini embeddings),
    vyhľadáva sa kosínovou podobnosťou; inak sa použije prekryv kľúčových slov.
    """

    def __init__(self, docs_dir: Path = DOCS_DIR, embedder: Optional[Embedder] = None):
        self.documents: List[Dict[str, str]] = self._load_documents(Path(docs_dir))
        self.embedder = embedder
        self._matrix: Optional[np.ndarray] = None

    @property
    def mode(self) -> str:
        return "embeddings" if self._matrix is not None else "keywords"

    @staticmethod
    def _load_documents(docs_dir: Path) -> List[Dict[str, str]]:
        documents = []
        for path in sorted(docs_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            sections = content.split("\n## ")
            title = sections[0].splitlines()[0].lstrip("# ").strip()
            for idx, sec in enumerate(sections):
                chunk = (sec if idx == 0 else f"## {sec}").strip()
                # Časť tvorená len nadpisom nenesie informáciu a zaberala by miesto vo výsledkoch
                if all(line.startswith("#") for line in chunk.splitlines() if line.strip()):
                    continue
                documents.append({"source": path.name, "title": title, "content": chunk})
        return documents

    def build_index(self) -> None:
        """Jedno dávkové volanie embeddings API. Pri chybe ostáva režim kľúčových slov."""
        if not self.embedder or not self.documents:
            return
        try:
            texts = [f"{d['title']}\n{d['content']}" for d in self.documents]
            vectors = np.array(self.embedder(texts, "RETRIEVAL_DOCUMENT"), dtype=np.float32)
            self._matrix = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            logger.info("RAG index built: %d chunks", len(self.documents))
        except Exception:
            logger.exception("Embedding index failed, falling back to keyword search")
            self._matrix = None

    def retrieve(self, query: str, top_k: int = 8, min_score: float = 0.65) -> List[Tuple[Dict[str, str], float]]:
        if not self.documents:
            return []
        if self._matrix is not None:
            try:
                queries = split_queries(query)
                vectors = self.embedder(queries, "RETRIEVAL_QUERY")  # jedno dávkové volanie
                return self.rank_multi(vectors, top_k, min_score)
            except Exception:
                logger.exception("Query embedding failed, using keyword search")
        return self._retrieve_keywords(query, top_k)

    def rank_multi(self, query_vectors, top_k: int, min_score: float, per_query: int = 3):
        """Rozklad dotazu: každá veta/otázka prispeje svojimi najlepšími zhodami.
        Pri e-maile s viacerými otázkami tak jedna téma neprebije ostatné."""
        best: Dict[int, float] = {}
        for vector in query_vectors:
            for idx, score in self._scores(vector, per_query, min_score):
                best[idx] = max(score, best.get(idx, 0.0))
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[i], s) for i, s in ranked]

    def _scores(self, query_vector, top_k: int, min_score: float):
        q = np.array(query_vector, dtype=np.float32)
        scores = self._matrix @ (q / np.linalg.norm(q))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] >= min_score]

    def rank(self, query_vector, top_k: int, min_score: float):
        """Zoradí časti podľa kosínovej podobnosti k už vypočítanému vektoru dotazu."""
        q = np.array(query_vector, dtype=np.float32)
        scores = self._matrix @ (q / np.linalg.norm(q))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in ranked if scores[i] >= min_score]

    def _retrieve_keywords(self, query: str, top_k: int):
        query_words = _tokens(query)
        results = []
        for doc in self.documents:
            overlap = len(query_words & _tokens(doc["title"] + " " + doc["content"]))
            if overlap:
                results.append((doc, overlap / max(len(query_words), 1)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

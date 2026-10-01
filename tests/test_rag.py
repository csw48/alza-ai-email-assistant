import numpy as np

from app.rag.knowledge_base import KnowledgeBase


def test_knowledge_base_loading():
    kb = KnowledgeBase()
    sources = {doc["source"] for doc in kb.documents}
    assert {"alza_drone_delivery.md", "alza_reclamation_policy.md", "alzaplus_vip_benefits.md"} <= sources
    assert all(doc["title"] and doc["content"] for doc in kb.documents)
    # žiadne časti tvorené len nadpisom
    assert not any(doc["content"].startswith("# ") and "\n" not in doc["content"] for doc in kb.documents)
    assert kb.mode == "keywords"


def test_keyword_retrieval_drone():
    kb = KnowledgeBase()
    doc, _ = kb.retrieve("Aká je maximálna váha balíka pre drone delivery?", top_k=1)[0]
    assert doc["source"] == "alza_drone_delivery.md"
    assert "2,5 kg" in doc["content"] or "2.5 kg" in doc["content"]


def test_keyword_retrieval_ignores_diacritics_and_punctuation():
    kb = KnowledgeBase()
    doc, _ = kb.retrieve("Mozem vymenit tovar okamzite cez AlzaBox?", top_k=1)[0]
    assert doc["source"] == "alza_reclamation_policy.md"
    assert "AlzaBox" in doc["content"]


def test_keyword_retrieval_handles_slovak_inflection():
    kb = KnowledgeBase()
    sources = [d["source"] for d, _ in kb.retrieve("Koľko stojí doručenie dronom s AlzaPlus+?", top_k=2)]
    assert "alza_drone_delivery.md" in sources


def _fake_embedder(texts, task_type):
    # Deterministický "embedding": 1 dimenzia na každý dokument podľa kľúčového slova
    keys = ["dron", "reklam", "vip"]
    # Posledné 2 dimenzie oddeľujú dokumenty od dotazov, aby dotaz bez kľúčového slova nemal zhodu
    tail = [0.01, 0.0] if task_type == "RETRIEVAL_DOCUMENT" else [0.0, 0.01]
    return [[1.0 if k in t.lower() else 0.0 for k in keys] + tail for t in texts]


def test_embedding_retrieval_ranks_by_cosine():
    kb = KnowledgeBase(embedder=_fake_embedder)
    kb.build_index()
    assert kb.mode == "embeddings"
    results = kb.retrieve("otázka k VIP členstvu", top_k=1, min_score=0.5)
    assert results[0][0]["source"] == "alzaplus_vip_benefits.md"
    assert np.isclose(results[0][1], 1.0, atol=0.01)


def test_embedding_min_score_filters_irrelevant():
    kb = KnowledgeBase(embedder=_fake_embedder)
    kb.build_index()
    assert kb.retrieve("počasie zajtra", top_k=2, min_score=0.5) == []


def test_embedding_failure_falls_back_to_keywords():
    def broken(texts, task_type):
        raise RuntimeError("API down")

    kb = KnowledgeBase(embedder=broken)
    kb.build_index()
    assert kb.mode == "keywords"
    assert kb.retrieve("drone delivery cena", top_k=1)[0][0]["source"] == "alza_drone_delivery.md"


def test_split_queries_keeps_full_text_and_questions():
    from app.rag.knowledge_base import split_queries

    parts = split_queries("Dobrý deň,\nkoľko stojí dron s AlzaPlus+? A môžem vymeniť kávovar cez AlzaBox?\nJán")
    assert parts[0].startswith("Dobrý deň")
    assert "koľko stojí dron s AlzaPlus+?" in parts
    assert "A môžem vymeniť kávovar cez AlzaBox?" in parts
    assert "Ján" not in parts  # krátke fragmenty (podpis) sa ignorujú


def test_multi_question_email_retrieves_all_topics():
    kb = KnowledgeBase(embedder=_fake_embedder)
    kb.build_index()
    # Celý text je dominantne o dronoch, ale druhá veta je o reklamácii
    results = kb.retrieve("Dron dron dron dron cena? Chcem reklamovať kávovar.", top_k=8, min_score=0.5)
    sources = {d["source"] for d, _ in results}
    assert "alza_drone_delivery.md" in sources
    assert "alza_reclamation_policy.md" in sources

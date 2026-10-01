"""Voliteľný end-to-end test proti skutočnému Gemini. Spustenie:
LIVE_GEMINI_API_KEY=... pytest tests/test_live_gemini.py
"""
import os

import pytest

from app.config import Settings
from app.rag.knowledge_base import KnowledgeBase
from app.services.gemini_service import GeminiService

LIVE_KEY = os.environ.get("LIVE_GEMINI_API_KEY")
pytestmark = pytest.mark.skipif(not LIVE_KEY, reason="LIVE_GEMINI_API_KEY nie je nastavený")


def test_live_reply_uses_knowledge_base():
    ai = GeminiService.from_settings(Settings(GEMINI_API_KEY=LIVE_KEY))
    kb = KnowledgeBase(embedder=ai.embed)
    kb.build_index()
    assert kb.mode == "embeddings"

    query = "Koľko stojí doručenie dronom, keď mám AlzaPlus+?"
    retrieved = kb.retrieve(query, top_k=2)
    context = "\n\n".join(d["content"] for d, _ in retrieved)
    reply = ai.generate_email_reply(
        sender="zakaznik@priklad.sk", subject="Drone delivery", body_text=query,
        attachments=[], knowledge_context=context,
    )
    assert reply.language == "sk"
    assert "1.99" in reply.reply or "1,99" in reply.reply

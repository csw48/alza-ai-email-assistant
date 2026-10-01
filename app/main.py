import html
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from app.config import settings
from app.idempotency import IdempotencyCache
from app.models.schemas import EmailAttachment, ProcessEmailRequest, ProcessEmailResponse
from app.rag.knowledge_base import KnowledgeBase
from app.services.gemini_service import AIServiceError, GeminiService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("alza_assistant")

REPLY_PREFIX = re.compile(r"^\s*(re|odp|aw|sv)\s*:", re.IGNORECASE)


def reply_subject(subject: str) -> str:
    subject = subject.strip() or "Váš dopyt"
    return subject if REPLY_PREFIX.match(subject) else f"Re: {subject}"


def text_to_html(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return "".join(f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in paragraphs)


def select_attachments(
    attachments: List[EmailAttachment], max_total_bytes: int
) -> Tuple[List[EmailAttachment], List[str]]:
    """Vráti (spracovateľné prílohy, názvy preskočených) – nepodporovaný typ alebo prekročený limit."""
    selected, skipped, total = [], [], 0
    for att in attachments:
        if not att.is_supported or total + att.size_bytes > max_total_bytes:
            skipped.append(att.filename)
            continue
        total += att.size_bytes
        selected.append(att)
    return selected, skipped


def create_app(kb: Optional[KnowledgeBase] = None, ai: Optional[GeminiService] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ai = ai or GeminiService.from_settings(settings)
        app.state.kb = kb or KnowledgeBase(embedder=app.state.ai.embed)
        app.state.kb.build_index()
        app.state.idempotency = IdempotencyCache()
        logger.info("Started: model=%s rag_mode=%s", settings.GEMINI_MODEL, app.state.kb.mode)
        yield

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Multimodálna AI mikroslužba pre Alza e-mailového asistenta s podporou RAG a príloh (PDF, audio, obrázky).",
        lifespan=lifespan,
    )

    def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
        key = (x_api_key or "").strip()
        if not key or not secrets.compare_digest(key.encode(), settings.N8N_API_KEY.encode()):
            # Hodnotu kľúča nelogujeme, len či prišiel a akú má dĺžku (očakávaná = 64)
            logger.warning("Auth failed: x-api-key present=%s length=%d", x_api_key is not None, len(key))
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing x-api-key")

    @app.get("/health")
    def health_check(request: Request):
        return {
            "status": "healthy",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "rag_documents_loaded": len(request.app.state.kb.documents),
            "rag_mode": request.app.state.kb.mode,
            "gemini_model": settings.GEMINI_MODEL,
        }

    # Synchrónny endpoint: FastAPI ho spustí v threadpoole, blokujúce volania Gemini
    # tak nezablokujú event loop.
    @app.post(
        f"{settings.API_V1_PREFIX}/process-email",
        response_model=ProcessEmailResponse,
        summary="Spracovanie prichádzajúceho e-mailu a vygenerovanie inteligentnej odpovede",
        dependencies=[Depends(verify_api_key)],
    )
    def process_email(payload: ProcessEmailRequest, request: Request):
        # Opakovaný request s rovnakým message_id (retry z n8n) dostane už vygenerovanú odpoveď
        response, duplicate = request.app.state.idempotency.run(
            payload.message_id, lambda: _process(payload, request.app.state.kb, request.app.state.ai)
        )
        if duplicate:
            logger.info("Duplicate message_id=%s served from cache", payload.message_id)
        return response

    def _process(payload: ProcessEmailRequest, kb: KnowledgeBase, ai: GeminiService) -> ProcessEmailResponse:
        start = time.perf_counter()

        # 1. RAG – vyhľadanie relevantných častí znalostnej bázy
        retrieved = kb.retrieve(
            f"{payload.subject}\n{payload.body_text}", top_k=settings.RAG_TOP_K, min_score=settings.RAG_MIN_SCORE
        )
        knowledge_context = "\n\n".join(f"[{doc['title']}]\n{doc['content']}" for doc, _ in retrieved)
        sources = sorted({doc["source"] for doc, _ in retrieved})

        # 2. Prílohy – len podporované typy v rámci limitu veľkosti
        attachments, skipped = select_attachments(
            payload.attachments, int(settings.MAX_ATTACHMENTS_TOTAL_MB * 1024 * 1024)
        )

        # 3. Generovanie odpovede. Pri zlyhaní vrátime 503 – n8n workflow skončí chybou
        #    a zákazníkovi neodíde žiadna vymyslená náhradná odpoveď.
        try:
            reply = ai.generate_email_reply(
                sender=payload.sender,
                subject=payload.subject,
                body_text=payload.body_text,
                attachments=attachments,
                knowledge_context=knowledge_context,
                conversation_history=payload.conversation_history,
                skipped_attachments=skipped,
            )
        except AIServiceError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI service unavailable")

        elapsed = round(time.perf_counter() - start, 2)
        email_age = None
        if payload.received_at:
            received = payload.received_at if payload.received_at.tzinfo else payload.received_at.replace(tzinfo=timezone.utc)
            email_age = round((datetime.now(timezone.utc) - received).total_seconds(), 1)
        logger.info(
            "Processed message_id=%s intent=%s needs_human=%s sources=%s attachments=%d skipped=%d time=%.2fs email_age=%ss",
            payload.message_id, reply.intent, reply.needs_human_review, sources, len(attachments), len(skipped), elapsed,
            email_age,
        )

        return ProcessEmailResponse(
            message_id=payload.message_id,
            thread_id=payload.thread_id,
            reply_subject=reply_subject(payload.subject),
            reply_body_text=reply.reply,
            reply_body_html=text_to_html(reply.reply),
            language=reply.language,
            intent=reply.intent,
            needs_human_review=reply.needs_human_review or bool(skipped),
            review_reason=reply.review_reason or ("Niektoré prílohy neboli spracované" if skipped else ""),
            attachments_processed=[a.filename for a in attachments],
            attachments_skipped=skipped,
            retrieved_knowledge_sources=sources,
            retrieval_mode=kb.mode,
            processing_time_seconds=elapsed,
            email_age_seconds=email_age,
        )

    return app


app = create_app()

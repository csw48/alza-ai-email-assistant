import base64
import binascii
import mimetypes
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# MIME typy, ktoré Gemini spracuje natívne. Ostatné prílohy sa preskočia
# a model dostane o nich len poznámku.
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/aac",
    "audio/flac",
    "audio/mp4",
    "audio/x-m4a",
    "text/plain",
}

# Rôzne klienty (iPhone, Outlook, Gmail) posielajú pre rovnaký formát rôzne MIME typy
MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/mp4a-latm": "audio/mp4",
    "audio/x-aac": "audio/aac",
}


class EmailAttachment(BaseModel):
    filename: str = Field(..., description="Názov súboru prílohy (napr. photo.jpg, invoice.pdf, voice.mp3)")
    mime_type: str = Field(..., description="MIME typ prílohy (napr. application/pdf, image/jpeg, audio/mpeg)")
    data_base64: str = Field(..., description="Base64 encoded obsah prílohy")

    @model_validator(mode="after")
    def normalize_mime(self) -> "EmailAttachment":
        mime = self.mime_type.split(";")[0].strip().lower()
        # Neznámy/generický typ -> odhad podľa prípony súboru
        if mime in ("", "application/octet-stream"):
            mime = mimetypes.guess_type(self.filename)[0] or mime
        self.mime_type = MIME_ALIASES.get(mime, mime)
        return self

    @field_validator("data_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("data_base64 nie je platný base64 reťazec")
        return v

    @property
    def is_supported(self) -> bool:
        return self.mime_type in SUPPORTED_MIME_TYPES

    def decoded(self) -> bytes:
        return base64.b64decode(self.data_base64)

    @property
    def size_bytes(self) -> int:
        # Veľkosť po dekódovaní bez samotného dekódovania
        return len(self.data_base64) * 3 // 4 - self.data_base64.count("=", -2)


class ProcessEmailRequest(BaseModel):
    message_id: str = Field(..., description="Unikátne ID správy (Message-ID)")
    thread_id: Optional[str] = Field(None, description="Thread ID v rámci Gmailu")
    sender: str = Field(..., description="E-mail odosielateľa (From)")
    recipient: str = Field(..., description="E-mail príjemcu (To)")
    subject: str = Field("", description="Predmet e-mailu")
    body_text: str = Field("", description="Text tela e-mailu (vrátane citovanej histórie vlákna)")
    attachments: List[EmailAttachment] = Field(default_factory=list, description="Zoznam priložených súborov")
    conversation_history: Optional[List[str]] = Field(default=None, description="Predchádzajúce správy vo vlákne (voliteľné)")
    received_at: Optional[datetime] = Field(None, description="Kedy e-mail prišiel do schránky (na meranie SLA)")


class AIReply(BaseModel):
    """Štruktúrovaný výstup modelu (Gemini response_schema)."""

    language: Literal["sk", "cs", "other"] = Field(..., description="Jazyk zákazníka")
    intent: Literal["reclamation", "delivery", "membership", "order_status", "other"] = Field(
        ..., description="Hlavná téma e-mailu"
    )
    reply: str = Field(..., description="Text odpovede zákazníkovi vrátane pozdravu a podpisu")
    attachments_summary: str = Field(..., description="Stručne, čo model zistil z príloh; prázdne ak žiadne neboli")
    needs_human_review: bool = Field(
        ..., description="True, ak odpoveď vyžaduje kontrolu človekom (chýbajúce fakty, sťažnosť, nejasná požiadavka)"
    )
    review_reason: str = Field(..., description="Dôvod, prečo je potrebná kontrola človekom; prázdne ak nie je potrebná")


class ProcessEmailResponse(BaseModel):
    message_id: str
    thread_id: Optional[str]
    reply_subject: str
    reply_body_text: str
    reply_body_html: str
    language: str
    intent: str
    needs_human_review: bool
    review_reason: str = ""
    attachments_processed: List[str] = Field(default_factory=list)
    attachments_skipped: List[str] = Field(default_factory=list)
    retrieved_knowledge_sources: List[str] = Field(default_factory=list)
    retrieval_mode: str
    processing_time_seconds: float
    # Čas od prijatia e-mailu do odoslania odpovede z API (polling n8n + spracovanie)
    email_age_seconds: Optional[float] = None

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Alza AI Email Assistant"
    VERSION: str = "1.1.0"
    API_V1_PREFIX: str = "/api/v1"

    # Gemini – na Cloud Run cez Vertex AI (service account, bez API kľúča),
    # lokálne stačí GEMINI_API_KEY z AI Studia.
    GOOGLE_GENAI_USE_VERTEXAI: bool = False
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_CLOUD_LOCATION: str = "global"  # gemini-3.6-flash je na Vertexe dostupný len v global
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.6-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_TEMPERATURE: float = 0.2
    # MINIMAL / LOW / MEDIUM / HIGH; prázdne = predvolené správanie modelu
    GEMINI_THINKING_LEVEL: Optional[str] = None

    # Zdieľaný tajný kľúč medzi n8n a touto službou (hlavička x-api-key).
    # Povinný – bez neho služba neštartuje.
    N8N_API_KEY: str

    # Limity príloh (Gemini inline dáta majú limit ~20 MB na request)
    MAX_ATTACHMENTS_TOTAL_MB: float = 18.0
    # Kalibrované na gemini-embedding-001 (Vertex, 2026-09): relevantné časti ~0.67–0.77,
    # otázka mimo znalostnej bázy max ~0.64. Pri e-mailoch s viacerými otázkami treba viac častí.
    RAG_TOP_K: int = 8
    RAG_MIN_SCORE: float = 0.65


settings = Settings()

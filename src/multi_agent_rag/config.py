"""Configuración leída de variables de entorno (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ / ".env")


@dataclass(frozen=True)
class Settings:
    llm_model: str
    llm_model_rapido: str
    db_path: Path
    ontologia_path: Path
    docs_dir: Path
    max_pasos_agente: int
    max_filas: int
    umbral_confianza: int
    top_k_documentos: int


def get_settings() -> Settings:
    llm_model = os.getenv("LLM_MODEL", "anthropic:claude-sonnet-5-5")
    return Settings(
        llm_model=llm_model,
        llm_model_rapido=os.getenv("LLM_MODEL_RAPIDO") or llm_model,
        db_path=Path(os.getenv("DB_PATH", RAIZ / "data" / "banco.db")),
        ontologia_path=Path(os.getenv("ONTOLOGIA_PATH", RAIZ / "data" / "ontologia.json")),
        docs_dir=Path(os.getenv("DOCS_DIR", RAIZ / "data" / "documentos")),
        max_pasos_agente=int(os.getenv("MAX_PASOS_AGENTE", "12")),
        max_filas=int(os.getenv("MAX_FILAS", "100")),
        umbral_confianza=int(os.getenv("UMBRAL_CONFIANZA", "80")),
        top_k_documentos=int(os.getenv("TOP_K_DOCUMENTOS", "3")),
    )

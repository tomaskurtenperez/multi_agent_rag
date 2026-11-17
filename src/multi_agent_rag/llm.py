"""Fábrica de modelos, independiente del proveedor (Anthropic, OpenAI, Azure OpenAI...)."""

from __future__ import annotations

from typing import Any


def crear_llm(spec: str) -> Any:
    """Crea un chat model a partir de "proveedor:modelo".

    Para Azure OpenAI, el modelo es el nombre del deployment.
    """
    from langchain.chat_models import init_chat_model

    proveedor, _, modelo = spec.partition(":")
    if not modelo:
        raise ValueError(f"Formato inválido '{spec}'. Usá 'proveedor:modelo', p. ej. 'openai:gpt-4.1'.")
    kwargs: dict[str, Any] = {}
    if proveedor == "azure_openai":
        kwargs["azure_deployment"] = modelo
    return init_chat_model(modelo, model_provider=proveedor, **kwargs)

"""
llm.py — loads the LLM (the "brain"). Switch with provider = "groq" or "gemini".
"""

from app import config


def get_llm(provider: str):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0,
            thinking_budget=0 if "flash" in config.GEMINI_MODEL else None,  # skip "thinking" = faster answers
        )

    from langchain_groq import ChatGroq

    return ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0)

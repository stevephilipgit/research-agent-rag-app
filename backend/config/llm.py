from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, DEFAULT_MODEL, LLM_TIMEOUT, LLM_TEMPERATURE

def get_llm(model_name=None):
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model_name or DEFAULT_MODEL,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
        max_retries=2
    )

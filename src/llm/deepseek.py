# src/llm/deepseek.py
from langchain_deepseek import ChatDeepSeek
from config import DEEPSEEK_API_KEY


def get_llm(temperature: float = 0.0) -> ChatDeepSeek:
    """Returns a configured DeepSeek instance."""
    return ChatDeepSeek(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        temperature=temperature,
    )

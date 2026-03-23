from typing import TypedDict, Optional, List, Annotated, Dict, Any
from operator import add


class HelpdeskState(TypedDict):
    consultation: str
    category: str
    response_rag: Optional[str]
    confidence: float
    sources: List[str]
    context_rag: Optional[str]
    required_human: bool
    response_human: Optional[str]
    response_final: Optional[str]
    history: Annotated[List[str], add]

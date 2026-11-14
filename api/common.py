"""Common utilities and schemas for API routers."""
import math
from typing import Any, Optional
from pydantic import BaseModel


def safe_value(v: Any) -> Any:
    """Replace NaN / Inf / -Inf with None so json.dumps never raises."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def safe_dict(d: dict) -> dict:
    return {k: safe_value(v) for k, v in d.items()}


# Backward compatibility aliases
_safe = safe_value
_safe_dict = safe_dict


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]
    session_id: Optional[str] = None


class DocumentRequest(BaseModel):
    text: str
    doc_type: str = "description"


class BikeRouteRequest(BaseModel):
    start: str
    end: str
    city: str = "Pittsburgh, PA"


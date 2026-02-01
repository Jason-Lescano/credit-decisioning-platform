from pydantic import BaseModel
from typing import Any, Dict, List


class ScoreRequest(BaseModel):
    features: Dict[str, Any]


class ScoreResponse(BaseModel):
    pd: float
    decision: str
    reasons: List[str]

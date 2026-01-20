from pydantic import BaseModel
from typing import List

class AnalysisResponse(BaseModel):
    summary: str
    detected_attributes: List[str]
    confidence_score: float

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class OCRMode(str, Enum):
    MARKDOWN = "markdown"
    OCR = "ocr"


class OCRResponse(BaseModel):
    markdown: str
    pages: int = 1
    success: bool = True
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

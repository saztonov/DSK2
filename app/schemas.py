from enum import Enum
from typing import Optional, List
from pydantic import BaseModel


class OCRMode(str, Enum):
    MARKDOWN = "markdown"
    OCR = "ocr"


class OCRResponse(BaseModel):
    markdown: str
    pages: int = 1
    success: bool = True
    error: Optional[str] = None


class GPUDevice(BaseModel):
    index: int
    name: str
    compute_capability: str
    total_memory_gb: float
    multi_processor_count: int


class GPUInfoResponse(BaseModel):
    cuda_available: bool
    cuda_version: Optional[str] = None
    pytorch_version: str
    device_count: int
    devices: List[GPUDevice] = []
    active_device: Optional[str] = None
    model_device: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: Optional[str] = None


class BatchOCRResponse(BaseModel):
    """Ответ для batch OCR запроса"""
    results: list[str]
    success: bool = True
    error: Optional[str] = None
    processed: int = 0
    failed: int = 0

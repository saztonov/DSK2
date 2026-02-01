import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from app.schemas import OCRMode, OCRResponse, HealthResponse, BatchOCRResponse, GPUInfoResponse, GPUDevice
from app.ocr_service import ocr_service, get_gpu_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from typing import List

SUPPORTED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/bmp", "image/tiff"
}
SUPPORTED_PDF_TYPE = "application/pdf"

# Максимальный размер batch (защита от OOM)
MAX_BATCH_SIZE = 16


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OCR service...")
    ocr_service.load_model()
    yield
    logger.info("Shutting down OCR service...")


app = FastAPI(
    title="DeepSeek-OCR-2 API",
    description="OCR API for text and table recognition using DeepSeek-OCR-2 model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if ocr_service.is_ready else "loading",
        model_loaded=ocr_service.is_ready,
        device=ocr_service.device
    )


@app.get("/gpu", response_model=GPUInfoResponse)
async def gpu_info():
    """Get detailed GPU information and current device status."""
    info = get_gpu_info()
    
    devices = [
        GPUDevice(
            index=d["index"],
            name=d["name"],
            compute_capability=d["compute_capability"],
            total_memory_gb=d["total_memory_gb"],
            multi_processor_count=d["multi_processor_count"]
        )
        for d in info.get("devices", [])
    ]
    
    return GPUInfoResponse(
        cuda_available=info["cuda_available"],
        cuda_version=info.get("cuda_version"),
        pytorch_version=torch.__version__,
        device_count=info["device_count"],
        devices=devices,
        active_device=ocr_service.device,
        model_device=ocr_service.device if ocr_service.is_ready else None
    )


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(..., description="Image or PDF file"),
    mode: OCRMode = Form(default=OCRMode.MARKDOWN, description="OCR mode"),
    page: int = Form(default=None, description="Single page to process (1-based)"),
    first_page: int = Form(default=None, description="First page to process (1-based)"),
    last_page: int = Form(default=None, description="Last page to process (1-based)")
):
    content_type = file.content_type or ""

    if content_type not in SUPPORTED_IMAGE_TYPES and content_type != SUPPORTED_PDF_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. "
                   f"Supported: {', '.join(SUPPORTED_IMAGE_TYPES | {SUPPORTED_PDF_TYPE})}"
        )

    try:
        file_bytes = await file.read()

        if content_type == SUPPORTED_PDF_TYPE:
            # If single page specified, use it for both first_page and last_page
            if page is not None:
                first_page = page
                last_page = page
            
            results = ocr_service.process_pdf(
                file_bytes, 
                mode,
                first_page=first_page,
                last_page=last_page
            )
            markdown = "\n\n---\n\n".join(results)
            pages = len(results)
        else:
            markdown = ocr_service.recognize_bytes(file_bytes, mode)
            pages = 1

        return OCRResponse(
            markdown=markdown,
            pages=pages,
            success=True
        )

    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(
    files: List[UploadFile] = File(..., description="List of image files"),
    mode: OCRMode = Form(default=OCRMode.MARKDOWN, description="OCR mode"),
):
    """
    Batch OCR endpoint - обработка нескольких изображений за один запрос.

    Уменьшает накладные расходы на HTTP, обрабатывая несколько изображений
    в одном запросе. Изображения обрабатываются последовательно на GPU.

    Args:
        files: Список файлов изображений (PNG, JPEG, WebP, BMP, TIFF)
        mode: Режим OCR - 'markdown' или 'ocr'

    Returns:
        BatchOCRResponse с результатами для каждого изображения
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum: {len(files)} > {MAX_BATCH_SIZE}"
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    # Проверяем типы файлов
    for i, file in enumerate(files):
        content_type = file.content_type or ""
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File {i + 1}: Unsupported type '{content_type}'. "
                       f"Supported: {', '.join(SUPPORTED_IMAGE_TYPES)}"
            )

    try:
        # Читаем все файлы
        images_bytes = []
        for file in files:
            file_bytes = await file.read()
            images_bytes.append(file_bytes)

        logger.info(f"Batch OCR: получено {len(images_bytes)} изображений")

        # Batch распознавание
        results, processed, failed = ocr_service.recognize_batch_bytes(
            images_bytes,
            mode
        )

        return BatchOCRResponse(
            results=results,
            success=failed == 0,
            processed=processed,
            failed=failed,
            error=f"{failed} images failed" if failed > 0 else None
        )

    except Exception as e:
        logger.exception("Batch OCR processing failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch OCR processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)

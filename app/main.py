import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from app.schemas import OCRMode, OCRResponse, HealthResponse
from app.ocr_service import ocr_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/bmp", "image/tiff"
}
SUPPORTED_PDF_TYPE = "application/pdf"


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
        model_loaded=ocr_service.is_ready
    )


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(..., description="Image or PDF file"),
    mode: OCRMode = Form(default=OCRMode.MARKDOWN, description="OCR mode")
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
            results = ocr_service.process_pdf(file_bytes, mode)
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


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)

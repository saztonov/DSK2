import io
import tempfile
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_bytes

from app.config import MODEL_NAME, BASE_SIZE, IMAGE_SIZE
from app.schemas import OCRMode

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load_model(self) -> None:
        logger.info(f"Loading model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)

        self._is_ready = True
        logger.info("Model loaded successfully")

    def _get_prompt(self, mode: OCRMode) -> str:
        if mode == OCRMode.MARKDOWN:
            return "<image>\n<|grounding|>Convert the document to markdown."
        return "<image>\nFree OCR."

    def recognize(self, image: Image.Image, mode: OCRMode = OCRMode.MARKDOWN) -> str:
        if not self._is_ready:
            raise RuntimeError("Model not loaded")

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "input.png"
            image.save(image_path, format="PNG")

            prompt = self._get_prompt(mode)

            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=tmp_dir,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=True,
                save_results=False
            )

            if isinstance(result, dict):
                return result.get("text", str(result))
            return str(result)

    def recognize_bytes(self, image_bytes: bytes, mode: OCRMode = OCRMode.MARKDOWN) -> str:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.recognize(image, mode)

    def process_pdf(self, pdf_bytes: bytes, mode: OCRMode = OCRMode.MARKDOWN) -> list[str]:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
        results = []

        for page in pages:
            if page.mode != "RGB":
                page = page.convert("RGB")
            result = self.recognize(page, mode)
            results.append(result)

        return results


ocr_service = OCRService()

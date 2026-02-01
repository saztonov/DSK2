import io
import base64
import logging
from typing import Optional

import httpx
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes

from app.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    VLLM_MODEL_NAME,
    VLLM_TIMEOUT,
    NGRAM_SIZE,
    WINDOW_SIZE,
    MAX_TOKENS,
)
from app.schemas import OCRMode

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def device(self) -> str:
        return "vllm-server" if self._is_ready else None

    def load_model(self) -> None:
        """Initialize OpenAI client for vLLM server."""
        logger.info(f"Connecting to vLLM server at: {VLLM_BASE_URL}")

        self._client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=httpx.Timeout(VLLM_TIMEOUT, connect=60.0),
        )

        # Check vLLM server health
        try:
            models = self._client.models.list()
            available_models = [m.id for m in models.data]
            logger.info(f"Available models on vLLM server: {available_models}")

            if VLLM_MODEL_NAME not in available_models and available_models:
                logger.warning(
                    f"Configured model '{VLLM_MODEL_NAME}' not in available models. "
                    f"Using first available: {available_models[0]}"
                )

            self._is_ready = True
            logger.info("Connected to vLLM server successfully")

        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            raise

    def _get_prompt(self, mode: OCRMode) -> str:
        if mode == OCRMode.MARKDOWN:
            return "<|grounding|>Convert the document to markdown."
        return "Free OCR."

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def recognize(self, image: Image.Image, mode: OCRMode = OCRMode.MARKDOWN) -> str:
        if not self._is_ready:
            raise RuntimeError("Service not connected to vLLM server")

        # Convert image to base64
        base64_image = self._image_to_base64(image)
        prompt = self._get_prompt(mode)

        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Call vLLM server with NGram parameters
        response = self._client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_body={
                "vllm_xargs": {
                    "ngram_size": NGRAM_SIZE,
                    "window_size": WINDOW_SIZE,
                    "whitelist_token_ids": [128821, 128822],  # <td>, </td>
                }
            },
        )

        return response.choices[0].message.content or ""

    def recognize_bytes(
        self, image_bytes: bytes, mode: OCRMode = OCRMode.MARKDOWN
    ) -> str:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.recognize(image, mode)

    def process_pdf(
        self,
        pdf_bytes: bytes,
        mode: OCRMode = OCRMode.MARKDOWN,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> list[str]:
        pages = convert_from_bytes(
            pdf_bytes, dpi=200, first_page=first_page, last_page=last_page
        )
        results = []

        for i, page in enumerate(pages):
            if page.mode != "RGB":
                page = page.convert("RGB")
            result = self.recognize(page, mode)
            results.append(result)
            logger.info(f"PDF: processed page {i + 1}/{len(pages)}")

        return results

    def recognize_batch(
        self, images: list[Image.Image], mode: OCRMode = OCRMode.MARKDOWN
    ) -> tuple[list[str], int, int]:
        """
        Batch recognition of multiple images.

        Processes images sequentially via vLLM server.

        Args:
            images: List of PIL images
            mode: OCR mode (markdown or ocr)

        Returns:
            Tuple (results, processed_count, failed_count)
        """
        if not self._is_ready:
            raise RuntimeError("Service not connected to vLLM server")

        results = []
        processed = 0
        failed = 0

        for i, image in enumerate(images):
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")

                result = self.recognize(image, mode)
                results.append(result)
                processed += 1
                logger.info(f"Batch: processed image {i + 1}/{len(images)}")

            except Exception as e:
                logger.error(f"Batch: error for image {i + 1}: {e}")
                results.append("")
                failed += 1

        logger.info(f"Batch completed: {processed} successful, {failed} failed")
        return results, processed, failed

    def recognize_batch_bytes(
        self, images_bytes: list[bytes], mode: OCRMode = OCRMode.MARKDOWN
    ) -> tuple[list[str], int, int]:
        """Batch recognition from bytes."""
        images = []
        for img_bytes in images_bytes:
            try:
                image = Image.open(io.BytesIO(img_bytes))
                images.append(image)
            except Exception as e:
                logger.error(f"Error opening image: {e}")
                images.append(None)

        # Filter None and remember indices
        valid_images = [(i, img) for i, img in enumerate(images) if img is not None]

        if not valid_images:
            return [], 0, len(images_bytes)

        # Recognize only valid images
        valid_results, processed, failed = self.recognize_batch(
            [img for _, img in valid_images], mode
        )

        # Restore order with empty strings for invalid
        results = [""] * len(images_bytes)
        for (orig_idx, _), result in zip(valid_images, valid_results):
            results[orig_idx] = result

        # Add failed from invalid images
        invalid_count = len(images_bytes) - len(valid_images)

        return results, processed, failed + invalid_count


ocr_service = OCRService()

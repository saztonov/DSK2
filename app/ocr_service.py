import io
import tempfile
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_bytes

from app.config import MODEL_NAME, BASE_SIZE, IMAGE_SIZE, DEVICE
from app.schemas import OCRMode

logger = logging.getLogger(__name__)


def get_gpu_info() -> dict:
    """Get detailed GPU information for logging."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "device_count": 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count
            }
            info["devices"].append(device_info)
    
    return info


def select_device(config_device: str) -> str:
    """Select the appropriate device based on configuration and availability."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif config_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available, falling back to CPU")
            return "cpu"
        return config_device
    else:
        return config_device


class OCRService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_ready = False
        self._device = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def device(self) -> str:
        return self._device

    def load_model(self) -> None:
        logger.info(f"Loading model: {MODEL_NAME}")
        
        # Log GPU information
        gpu_info = get_gpu_info()
        logger.info(f"CUDA available: {gpu_info['cuda_available']}")
        if gpu_info['cuda_available']:
            logger.info(f"CUDA version: {gpu_info['cuda_version']}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"Number of GPUs: {gpu_info['device_count']}")
            for dev in gpu_info['devices']:
                logger.info(
                    f"  GPU {dev['index']}: {dev['name']} "
                    f"(Compute Capability: {dev['compute_capability']}, "
                    f"Memory: {dev['total_memory_gb']} GB)"
                )
        else:
            logger.warning("CUDA is NOT available - running on CPU will be slow!")

        # Select device
        self._device = select_device(DEVICE)
        logger.info(f"Selected device: {self._device} (configured: {DEVICE})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            attn_implementation='eager',  # Use eager instead of flash_attention_2
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Move model to selected device
        self.model = self.model.eval().to(self._device).to(torch.bfloat16)
        
        # Verify model is on correct device
        if hasattr(self.model, 'device'):
            logger.info(f"Model loaded on device: {self.model.device}")
        
        self._is_ready = True
        logger.info(f"Model loaded successfully on {self._device}")

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
                save_results=True
            )

            # Log all files in output directory for debugging
            all_files = list(Path(tmp_dir).rglob("*"))
            logger.info(f"Files in output dir: {[str(f) for f in all_files]}")

            # Try to read from saved result file if result is None or empty
            if not result:
                # Look for result.mmd file (MultiMarkdown format)
                mmd_file = Path(tmp_dir) / "result.mmd"
                if mmd_file.exists():
                    content = mmd_file.read_text(encoding="utf-8")
                    if content.strip():
                        logger.info(f"Found result in: {mmd_file}")
                        return content
                
                # Look for markdown files (.md and .mmd)
                for md_file in Path(tmp_dir).rglob("*.mmd"):
                    content = md_file.read_text(encoding="utf-8")
                    if content.strip():
                        logger.info(f"Found markdown in: {md_file}")
                        return content
                
                for md_file in Path(tmp_dir).rglob("*.md"):
                    content = md_file.read_text(encoding="utf-8")
                    if content.strip():
                        logger.info(f"Found markdown in: {md_file}")
                        return content
                
                # Look for text files
                for txt_file in Path(tmp_dir).rglob("*.txt"):
                    content = txt_file.read_text(encoding="utf-8")
                    if content.strip():
                        logger.info(f"Found text in: {txt_file}")
                        return content

            if isinstance(result, dict):
                return result.get("text", str(result))
            return str(result) if result else ""

    def recognize_bytes(self, image_bytes: bytes, mode: OCRMode = OCRMode.MARKDOWN) -> str:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.recognize(image, mode)

    def process_pdf(
        self,
        pdf_bytes: bytes,
        mode: OCRMode = OCRMode.MARKDOWN,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> list[str]:
        pages = convert_from_bytes(
            pdf_bytes,
            dpi=200,
            first_page=first_page,
            last_page=last_page
        )
        results = []

        for page in pages:
            if page.mode != "RGB":
                page = page.convert("RGB")
            result = self.recognize(page, mode)
            results.append(result)

        return results

    def recognize_batch(
        self,
        images: list[Image.Image],
        mode: OCRMode = OCRMode.MARKDOWN
    ) -> tuple[list[str], int, int]:
        """
        Batch распознавание нескольких изображений.

        Обрабатывает изображения последовательно, но за один HTTP запрос,
        что уменьшает накладные расходы на сеть.

        Args:
            images: Список PIL изображений
            mode: Режим OCR (markdown или ocr)

        Returns:
            Tuple (results, processed_count, failed_count)
        """
        if not self._is_ready:
            raise RuntimeError("Model not loaded")

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
                logger.info(f"Batch: обработано изображение {i + 1}/{len(images)}")

            except Exception as e:
                logger.error(f"Batch: ошибка для изображения {i + 1}: {e}")
                results.append("")  # Пустой результат для failed
                failed += 1

        logger.info(f"Batch завершён: {processed} успешно, {failed} ошибок")
        return results, processed, failed

    def recognize_batch_bytes(
        self,
        images_bytes: list[bytes],
        mode: OCRMode = OCRMode.MARKDOWN
    ) -> tuple[list[str], int, int]:
        """Batch распознавание из bytes."""
        images = []
        for img_bytes in images_bytes:
            try:
                image = Image.open(io.BytesIO(img_bytes))
                images.append(image)
            except Exception as e:
                logger.error(f"Ошибка открытия изображения: {e}")
                images.append(None)

        # Фильтруем None и запоминаем индексы
        valid_images = [(i, img) for i, img in enumerate(images) if img is not None]

        if not valid_images:
            return [], 0, len(images_bytes)

        # Распознаём только валидные
        valid_results, processed, failed = self.recognize_batch(
            [img for _, img in valid_images],
            mode
        )

        # Восстанавливаем порядок с пустыми строками для invalid
        results = [""] * len(images_bytes)
        for (orig_idx, _), result in zip(valid_images, valid_results):
            results[orig_idx] = result

        # Добавляем failed от невалидных изображений
        invalid_count = len(images_bytes) - len(valid_images)

        return results, processed, failed + invalid_count


ocr_service = OCRService()

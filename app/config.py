import os

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "768"))
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "markdown")

# Device configuration: "auto", "cuda", "cuda:0", "cuda:1", "cpu"
DEVICE = os.getenv("DEVICE", "auto")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

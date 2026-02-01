import os

# vLLM Server Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm-server:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "3600"))

# OCR Configuration
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "markdown")

# NGram parameters for DeepSeek-OCR-2
NGRAM_SIZE = int(os.getenv("NGRAM_SIZE", "30"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "90"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# FastAPI Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

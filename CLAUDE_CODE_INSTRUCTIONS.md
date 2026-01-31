# DeepSeek-OCR-2 API - Инструкция для Claude Code

## Описание сервиса

OCR-сервис на базе модели **DeepSeek-OCR-2** для распознавания текста и таблиц из изображений и PDF-документов. Сервис возвращает результат в формате Markdown.

## Базовый URL

```
http://localhost:8001
```

> При локальном запуске через docker-compose порт `8001` маппится на внутренний `8000`.

---

## Endpoints

### 1. Health Check

Проверка состояния сервиса и загрузки модели.

**Запрос:**
```http
GET /health
```

**Ответ (JSON):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `status` | `string` | `"ok"` - сервис готов, `"loading"` - модель загружается |
| `model_loaded` | `boolean` | `true` если модель загружена и готова к работе |

**Пример curl:**
```bash
curl http://localhost:8001/health
```

---

### 2. OCR - Распознавание документа

Основной endpoint для распознавания текста.

**Запрос:**
```http
POST /ocr
Content-Type: multipart/form-data
```

**Параметры формы:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `file` | `file` | ✅ Да | Изображение или PDF-файл |
| `mode` | `string` | ❌ Нет | Режим OCR: `"markdown"` (по умолчанию) или `"ocr"` |

**Поддерживаемые форматы файлов:**
- Изображения: `image/png`, `image/jpeg`, `image/jpg`, `image/webp`, `image/bmp`, `image/tiff`
- Документы: `application/pdf`

**Режимы OCR:**

| Режим | Описание |
|-------|----------|
| `markdown` | Конвертирует документ в структурированный Markdown с сохранением таблиц и форматирования |
| `ocr` | Свободное распознавание текста без структурирования |

**Успешный ответ (JSON):**
```json
{
  "markdown": "# Заголовок документа\n\nТекст документа...\n\n| Колонка 1 | Колонка 2 |\n|-----------|-----------|",
  "pages": 1,
  "success": true,
  "error": null
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `markdown` | `string` | Распознанный текст в формате Markdown |
| `pages` | `integer` | Количество обработанных страниц |
| `success` | `boolean` | `true` при успешной обработке |
| `error` | `string \| null` | Сообщение об ошибке или `null` |

**Ответ при ошибке:**
```json
{
  "detail": "Unsupported file type: application/octet-stream. Supported: image/png, image/jpeg, ..."
}
```

HTTP коды:
- `200` - Успех
- `400` - Неподдерживаемый формат файла
- `500` - Ошибка обработки

---

## Примеры использования

### curl - Распознавание изображения

```bash
curl -X POST "http://localhost:8001/ocr" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.png" \
  -F "mode=markdown"
```

### curl - Распознавание PDF

```bash
curl -X POST "http://localhost:8001/ocr" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "mode=markdown"
```

### Python - requests

```python
import requests

url = "http://localhost:8001/ocr"

# Для изображения
with open("document.png", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("document.png", f, "image/png")},
        data={"mode": "markdown"}
    )

result = response.json()
print(result["markdown"])
print(f"Страниц: {result['pages']}")
```

### Python - httpx (async)

```python
import httpx

async def ocr_document(file_path: str, mode: str = "markdown") -> dict:
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                "http://localhost:8001/ocr",
                files={"file": f},
                data={"mode": mode}
            )
        return response.json()
```

### JavaScript - fetch

```javascript
async function ocrDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mode', 'markdown');

  const response = await fetch('http://localhost:8001/ocr', {
    method: 'POST',
    body: formData
  });

  return await response.json();
}
```

---

## Особенности обработки PDF

- Каждая страница PDF конвертируется в изображение с DPI=200
- Результаты страниц объединяются через разделитель `\n\n---\n\n`
- Поле `pages` содержит общее количество обработанных страниц

**Пример ответа для многостраничного PDF:**
```json
{
  "markdown": "# Страница 1\nТекст первой страницы...\n\n---\n\n# Страница 2\nТекст второй страницы...",
  "pages": 2,
  "success": true,
  "error": null
}
```

---

## Запуск сервиса

### Docker Compose (рекомендуется)

```bash
docker-compose up -d
```

Требования:
- Docker с поддержкой NVIDIA GPU
- nvidia-container-toolkit

### Проверка готовности

После запуска модель загружается ~2-3 минуты. Проверяйте готовность через `/health`:

```bash
# Ждать пока model_loaded станет true
while true; do
  status=$(curl -s http://localhost:8001/health | jq -r '.model_loaded')
  if [ "$status" = "true" ]; then
    echo "Сервис готов!"
    break
  fi
  echo "Загрузка модели..."
  sleep 10
done
```

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MODEL_NAME` | `deepseek-ai/DeepSeek-OCR-2` | Имя модели на HuggingFace |
| `BASE_SIZE` | `1024` | Базовый размер для обработки |
| `IMAGE_SIZE` | `768` | Размер изображения |
| `DEFAULT_MODE` | `markdown` | Режим OCR по умолчанию |
| `HOST` | `0.0.0.0` | Хост для запуска |
| `PORT` | `8000` | Порт сервиса |

---

## OpenAPI / Swagger

Документация API доступна по адресам:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- OpenAPI JSON: `http://localhost:8001/openapi.json`

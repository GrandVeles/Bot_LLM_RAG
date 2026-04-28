# 🤖 Локальный RAG-бот технической поддержки

Полностью офлайн-система вопрос-ответ по внутренней документации.  
Данные **никуда не отправляются** — всё работает локально.

---

## Архитектура

```
Документы (TXT/PDF/DOCX/PPTX/MD)
        │
        ▼
 DirectoryLoader → TextSplitter (500 симв., overlap 50)
        │
        ▼
 SentenceTransformer (all-MiniLM-L6-v2)  →  ChromaDB (векторы на диске)
                                                   │
                                              Similarity Search (top-4)
                                                   │
                                         RAG Prompt + Ollama (LLM)
                                                   │
                                            Ответ + Источники
                                                   │
                                          Gradio Web UI (localhost)
```

---

## Установка

### 1. Python-зависимости

```bash
# Создайте виртуальное окружение
python -m venv .venv

# Активируйте его
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установите зависимости
pip install -r requirements.txt

# (опционально) CPU-only PyTorch — меньше размер загрузки:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **Первый запуск:** модель `all-MiniLM-L6-v2` (~90 MB) скачается автоматически  
> в `~/.cache/huggingface/hub/` и больше интернет не нужен.

---

### 2. Установка Ollama

#### Windows
1. Скачайте установщик: https://ollama.com/download/windows  
2. Запустите `OllamaSetup.exe`, следуйте инструкциям.

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

---

### 3. Скачивание модели LLM

```bash
# Запустите Ollama-сервер (если ещё не запущен)
ollama serve

# В другом терминале скачайте модель (нужен интернет только один раз)
ollama pull mistral:7b-instruct      # ~4.1 GB, рекомендуется
# или
ollama pull llama3:8b                # ~4.7 GB
# или лёгкая модель:
ollama pull qwen2:1.5b               # ~0.9 GB
```

После скачивания модели доступны **офлайн**.

---

### 4. Структура проекта

```
local_support_bot/
├── data/                 ← Положите сюда ваши документы
│   ├── manual.pdf
│   ├── faq.md
│   └── guide.docx
├── chroma_db/            ← Создаётся автоматически
├── app.py                ← Основной скрипт
├── requirements.txt
└── README.md
```

---

## Запуск

```bash
# 1. Убедитесь, что Ollama запущена:
ollama serve

# 2. Положите документы в папку data/

# 3. Запустите бота:
python app.py

# Первый запуск — автоматически проиндексирует документы
# Браузер откроется на http://127.0.0.1:7860
```

### Дополнительные параметры

```bash
# Принудительно переиндексировать (после добавления новых документов)
python app.py --reindex

# Использовать другую модель
python app.py --model llama3:8b

# Другая папка с документами
python app.py --docs-dir /path/to/my/docs

# Другой порт
python app.py --port 8080

# Публичная ссылка (через ngrok, нужен интернет)
python app.py --share
```

### Переменные окружения

Все параметры можно задать через env-переменные (префикс `BOT_`):

```bash
export BOT_OLLAMA_MODEL=llama3:8b
export BOT_CHUNK_SIZE=800
export BOT_RETRIEVER_K=6
python app.py
```

---

## Поддерживаемые форматы документов

| Формат | Расширения |
|--------|-----------|
| Текст | `.txt` |
| Markdown | `.md` |
| PDF | `.pdf` |
| Word | `.doc`, `.docx` |
| PowerPoint | `.ppt`, `.pptx` |

---

## Устранение неполадок

| Проблема | Решение |
|---------|---------|
| `Connection refused` к Ollama | Запустите `ollama serve` в отдельном терминале |
| Модель не найдена | Выполните `ollama pull mistral:7b-instruct` |
| `ModuleNotFoundError` | Активируйте venv и выполните `pip install -r requirements.txt` |
| Пустая база знаний | Добавьте файлы в `data/` и запустите `python app.py --reindex` |
| Медленные ответы | Переключитесь на меньшую модель: `--model qwen2:1.5b` |
| Ошибки DOCX/PPTX | Установите `pip install "unstructured[docx,pptx]"` |

---

## Минимальные системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| RAM | 8 GB | 16 GB |
| Дисковое пространство | 6 GB | 12 GB |
| CPU | 4 ядра | 8 ядер |
| GPU | не нужна | NVIDIA (ускорит LLM) |
| Python | 3.10 | 3.11+ |

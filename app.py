"""
Local RAG Technical Support Bot
================================
Полностью локальная система технической поддержки на базе LLM + RAG.

Стек:
  - Ollama  — локальный LLM (mistral / llama3 / любой другой)
  - ChromaDB — векторное хранилище
  - SentenceTransformers — эмбеддинги (all-MiniLM-L6-v2)
  - LangChain — оркестрация RAG-пайплайна (LCEL)
  - Gradio   — веб-интерфейс

Запуск:
  python app.py
  python app.py --reindex        # принудительная переиндексация
  python app.py --model llama3:8b
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# ─── LangChain (0.3+ / LCEL) ─────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import gradio as gr

# ──────────────────────────────────────────────────────────────────────────────
# Логгирование
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("support_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SupportBot")


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

class BotConfig(BaseSettings):
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="gemma2:2b")
    ollama_timeout: int = Field(default=120)

    docs_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path("chroma_db"))
    collection_name: str = Field(default="support_docs")

    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    retriever_k: int = Field(default=4)

    gradio_host: str = Field(default="127.0.0.1")
    gradio_port: int = Field(default=7860)
    gradio_share: bool = Field(default=False)

    @field_validator("docs_dir", "chroma_dir", mode="before")
    @classmethod
    def make_path(cls, v: str | Path) -> Path:
        return Path(v)

    model_config = {"env_prefix": "BOT_", "extra": "ignore"}


# ──────────────────────────────────────────────────────────────────────────────
# Загрузчики документов
# ──────────────────────────────────────────────────────────────────────────────

LOADER_MAP: dict[str, type] = {
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".pdf":  PyPDFLoader,
    ".doc":  UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".ppt":  UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
}


def _load_single_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        logger.warning("Неизвестный тип файла, пропускаем: %s", file_path)
        return []
    try:
        kwargs: dict = {}
        if loader_cls is TextLoader:
            kwargs["encoding"] = "utf-8"
        docs = loader_cls(str(file_path), **kwargs).load()
        for doc in docs:
            doc.metadata.setdefault("source", str(file_path))
        return docs
    except Exception as exc:
        logger.error("Ошибка загрузки %s: %s", file_path, exc)
        return []


def check_ollama(base_url: str, model: str, timeout: int = 10) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        name_base = model.split(":")[0]
        found = any(name_base in m for m in available)
        if not found:
            logger.warning("Модель '%s' не найдена. Доступные: %s", model, available)
        return found
    except requests.ConnectionError:
        logger.error("Ollama не отвечает на %s. Запустите: ollama serve", base_url)
        return False
    except Exception as exc:
        logger.error("Ошибка проверки Ollama: %s", exc)
        return False


def get_ollama_models(base_url: str) -> list[str]:
    """Получить список установленных моделей из Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# RAG Prompt
# ──────────────────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Ты — ассистент технической поддержки. Отвечай строго на основе предоставленного контекста.
Если информации недостаточно — скажи об этом прямо.
Отвечай на том же языке, на котором задан вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:""",
)


# ──────────────────────────────────────────────────────────────────────────────
# SupportBot
# ──────────────────────────────────────────────────────────────────────────────

class SupportBot:
    def __init__(self, config: BotConfig) -> None:
        self.cfg = config
        self._vectorstore: Optional[Chroma] = None
        self._chain = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._retrieved_docs: list[Document] = []

        self.cfg.docs_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.chroma_dir.mkdir(parents=True, exist_ok=True)

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info("Загружаем модель эмбеддингов: %s", self.cfg.embedding_model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.cfg.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    # ── Загрузка документов ──────────────────────────────────────────────────

    def load_documents(self) -> list[Document]:
        docs_dir = self.cfg.docs_dir
        supported = set(LOADER_MAP.keys())
        files = [f for f in docs_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported]

        if not files:
            logger.warning("Нет документов в %s", docs_dir)
            return []

        logger.info("Найдено файлов: %d", len(files))
        all_docs: list[Document] = []
        for fp in files:
            logger.info("  ↳ %s", fp.name)
            all_docs.extend(_load_single_file(fp))
        logger.info("Загружено страниц/секций: %d", len(all_docs))
        return all_docs

    # ── Чанкинг ─────────────────────────────────────────────────────────────

    def split_documents(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info("Чанков: %d (размер=%d, overlap=%d)",
                    len(chunks), self.cfg.chunk_size, self.cfg.chunk_overlap)
        return chunks

    # ── Векторное хранилище ──────────────────────────────────────────────────

    def create_vectorstore(self, chunks: list[Document]) -> Chroma:
        logger.info("Создаём ChromaDB в %s …", self.cfg.chroma_dir)
        t = time.time()
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            persist_directory=str(self.cfg.chroma_dir),
            collection_name=self.cfg.collection_name,
        )
        logger.info("ChromaDB готова за %.1f сек.", time.time() - t)
        self._vectorstore = vs
        self._chain = None  # сбросить цепочку
        return vs

    def load_vectorstore(self) -> Optional[Chroma]:
        db_file = self.cfg.chroma_dir / "chroma.sqlite3"
        if not db_file.exists():
            logger.info("База не найдена — нужна индексация.")
            return None
        try:
            vs = Chroma(
                persist_directory=str(self.cfg.chroma_dir),
                embedding_function=self._get_embeddings(),
                collection_name=self.cfg.collection_name,
            )
            logger.info("ChromaDB загружена: %d векторов.", vs._collection.count())
            self._vectorstore = vs
            return vs
        except Exception as exc:
            logger.error("Не удалось загрузить ChromaDB: %s", exc)
            return None

    # ── LCEL-цепочка ────────────────────────────────────────────────────────

    def _build_chain(self) -> None:
        """Собрать RAG-цепочку через LCEL (LangChain 0.3+)."""
        if self._vectorstore is None:
            raise RuntimeError("Векторное хранилище не инициализировано.")

        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.cfg.retriever_k},
        )
        llm = OllamaLLM(
            base_url=self.cfg.ollama_base_url,
            model=self.cfg.ollama_model,
            timeout=self.cfg.ollama_timeout,
        )

        # Сохраняем документы в атрибут для последующего отображения источников
        def retrieve_and_store(question: str) -> str:
            self._retrieved_docs = retriever.invoke(question)
            return "\n\n".join(d.page_content for d in self._retrieved_docs)

        self._chain = (
            {"context": retrieve_and_store, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        logger.info("LCEL-цепочка готова (модель: %s).", self.cfg.ollama_model)

    # ── Вопрос → ответ ───────────────────────────────────────────────────────

    def ask(self, question: str) -> tuple[str, list[dict]]:
        if not question.strip():
            return "Пожалуйста, введите вопрос.", []

        if self._chain is None:
            self._build_chain()

        logger.info("Вопрос: %s", question.strip())
        t = time.time()
        try:
            answer: str = self._chain.invoke(question)
        except Exception as exc:
            logger.error("Ошибка генерации: %s", exc)
            return f"❌ Ошибка: {exc}", []

        logger.info("Ответ за %.1f сек.", time.time() - t)

        # Дедупликация источников
        seen: set[str] = set()
        sources: list[dict] = []
        for doc in self._retrieved_docs:
            key = (doc.metadata.get("source", ""), doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "—"),
                    "content": doc.page_content.strip()[:300],
                })
        return answer.strip(), sources

    # ── Смена модели ─────────────────────────────────────────────────────────

    def switch_model(self, model_name: str) -> str:
        """Переключиться на другую модель Ollama."""
        if model_name == self.cfg.ollama_model:
            return f"Модель '{model_name}' уже активна."
        ok = check_ollama(self.cfg.ollama_base_url, model_name)
        if not ok:
            return f"❌ Модель '{model_name}' недоступна."
        self.cfg.ollama_model = model_name
        self._chain = None  # сбросить цепочку — пересоберётся при следующем вопросе
        logger.info("Модель переключена на: %s", model_name)
        return f"✅ Активная модель: **{model_name}**"

    # ── Инициализация ────────────────────────────────────────────────────────

    def initialize(self, force_reindex: bool = False) -> bool:
        vs = None if force_reindex else self.load_vectorstore()

        if vs is None:
            docs = self.load_documents()
            if docs:
                chunks = self.split_documents(docs)
                self.create_vectorstore(chunks)
            else:
                logger.warning("Нет документов. Добавьте файлы в '%s'.", self.cfg.docs_dir)

        ollama_ok = check_ollama(self.cfg.ollama_base_url, self.cfg.ollama_model)
        if not ollama_ok:
            logger.warning("Ollama недоступна — ответы будут недоступны.")
        return ollama_ok and self._vectorstore is not None


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_Источники не найдены._"
    lines = []
    for i, s in enumerate(sources, 1):
        fname = Path(s["source"]).name
        page = f", стр. {s['page']}" if s["page"] != "—" else ""
        lines.append(f"**[{i}] {fname}{page}**\n> {s['content']}")
    return "\n\n".join(lines)


def build_gradio_app(bot: SupportBot) -> gr.Blocks:

    def on_ask(question: str, history: list) -> tuple[list, str]:
        if not question.strip():
            return history, ""
        answer, sources = bot.ask(question)
        history = history or []
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, _format_sources(sources)

    def on_model_change(model_name: str) -> str:
        if not model_name:
            return ""
        return bot.switch_model(model_name)

    def refresh_models() -> gr.Dropdown:
        models = get_ollama_models(bot.cfg.ollama_base_url)
        return gr.Dropdown(choices=models, value=bot.cfg.ollama_model)

    def on_reindex() -> str:
        try:
            docs = bot.load_documents()
            if not docs:
                return "⚠️ Нет документов в папке data/."
            chunks = bot.split_documents(docs)
            bot.create_vectorstore(chunks)
            return f"✅ Переиндексировано: {len(chunks)} чанков."
        except Exception as exc:
            return f"❌ Ошибка: {exc}"

    with gr.Blocks(title="🤖 Локальный бот техподдержки") as demo:

        gr.Markdown(f"""
# 🤖 Локальный бот технической поддержки
> **Модель:** `{bot.cfg.ollama_model}` &nbsp;|&nbsp; **Документы:** `{bot.cfg.docs_dir}/`
""")

        # ── Выбор модели ──────────────────────────────────────────────────────
        with gr.Row():
            initial_models = get_ollama_models(bot.cfg.ollama_base_url)
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                value=bot.cfg.ollama_model if bot.cfg.ollama_model in initial_models else (initial_models[0] if initial_models else None),
                label="🤖 Активная модель Ollama",
                interactive=True,
                scale=4,
            )
            refresh_btn = gr.Button("🔃 Обновить список", scale=1)
            model_status = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Диалог", height=420)
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="Введите вопрос…",
                        label="Ваш вопрос", lines=2, scale=5,
                    )
                    ask_btn = gr.Button("💬 Спросить", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("🗑 Очистить")
                    reindex_btn = gr.Button("🔄 Переиндексировать")
                reindex_status = gr.Markdown("")

            with gr.Column(scale=2):
                sources_box = gr.Markdown(
                    value="_Источники появятся после первого вопроса._",
                    label="📚 Источники",
                )

        model_dropdown.change(on_model_change, inputs=model_dropdown, outputs=model_status)
        refresh_btn.click(refresh_models, outputs=model_dropdown)

        ask_btn.click(on_ask, [question_box, chatbot], [chatbot, sources_box]).then(
            lambda: "", outputs=question_box
        )
        question_box.submit(on_ask, [question_box, chatbot], [chatbot, sources_box]).then(
            lambda: "", outputs=question_box
        )
        clear_btn.click(lambda: ([], "_Источники появятся после первого вопроса._"), outputs=[chatbot, sources_box])  # noqa
        reindex_btn.click(on_reindex, outputs=reindex_status)

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Локальный RAG-бот техподдержки")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--docs-dir", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BotConfig()
    if args.model:    cfg.ollama_model = args.model
    if args.docs_dir: cfg.docs_dir = Path(args.docs_dir)
    if args.port:     cfg.gradio_port = args.port
    if args.share:    cfg.gradio_share = True

    logger.info("=" * 55)
    logger.info("  RAG-бот техподдержки | модель: %s", cfg.ollama_model)
    logger.info("  Документы: %s", cfg.docs_dir.resolve())
    logger.info("=" * 55)

    bot = SupportBot(config=cfg)
    bot.initialize(force_reindex=args.reindex)

    demo = build_gradio_app(bot)
    demo.launch(
        server_name=cfg.gradio_host,
        server_port=cfg.gradio_port,
        share=cfg.gradio_share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )


if __name__ == "__main__":
    main()

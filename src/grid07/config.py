

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR : Path = Path(__file__).resolve().parents[2]   # project root: grid07/
LOGS_DIR : Path = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE : Path = LOGS_DIR / "execution_log.md"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# Two handlers run simultaneously:
#   • Console  → INFO+  clean one-liner per event (visible while running)
#   • Markdown → DEBUG+ full detail written to logs/execution_log.md
#
# Phase section headers are triggered by logging at WARNING level with a
# message that starts with "##", e.g.:
#     log.warning("## Phase 1: Routing")
# This writes a proper markdown heading to the file, plain text to console.
# ══════════════════════════════════════════════════════════════════════════════

class MarkdownFileHandler(logging.FileHandler):
    """
    Custom FileHandler that writes to execution_log.md.

    Rules:
      - Messages starting with "##" are written as markdown headings (no prefix).
      - All other messages are written as plain lines.
      - A blank line is inserted before every "##" heading for readability.
    """

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("##"):
            # write as clean markdown heading — no timestamp prefix
            self.stream.write(f"\n{msg}\n\n")
        else:
            formatted = self.format(record)
            self.stream.write(f"{formatted}\n")
        self.flush()


class _ConsoleSafeFormatter(logging.Formatter):
    """
    Console formatter that strips leading '##' from section header messages
    so they don't look broken in the terminal.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if record.getMessage().startswith("##"):
            # Replace the raw message with a clean separator line
            return f"\n{'─' * 60}\n {record.getMessage().lstrip('# ').upper()}\n{'─' * 60}"
        return formatted


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Guard: don't add duplicate handlers on re-import
    if root.handlers:
        return

    # ── Silence noisy third-party loggers ─────────────────────────────────────
    for lib in ("httpx", "httpcore", "huggingface_hub",
                "sentence_transformers", "transformers", "filelock"):
        logging.getLogger(lib).setLevel(logging.ERROR)

    # ── Handler 1: Console ────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        _ConsoleSafeFormatter("%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    # ── Handler 2: Markdown file ──────────────────────────────────────────────
    # Only write grid07.* loggers to the markdown file — no third-party noise
    file_handler = MarkdownFileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
    file_handler.addFilter(lambda r: r.name.startswith(("grid07", "__main__", "##")))
    root.addHandler(file_handler)

    # Seed the markdown file with its top-level heading
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# Execution Logs\n\n")


_setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Factory for named loggers.

    Convention — use __name__ so log lines show the module:
        log = get_logger(__name__)
        log.info("router initialised")
        # → INFO     │ grid07.router                  │ router initialised
    """
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# All tuneable values live here. Change behaviour by editing .env, not code.
# ══════════════════════════════════════════════════════════════════════════════

class Settings:
    """
    Application-wide configuration loaded from environment variables.
    Sensible defaults are provided for every non-secret value so the
    project runs out-of-the-box once GROQ_API_KEY is supplied.
    """

    # ── LLM provider (Groq) ───────────────────────────────────────────────────
    GROQ_API_KEY : str   = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL    : str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Phase 1 · Router ──────────────────────────────────────────────────────
    # Cosine similarity cutoff for bot matching.
    # TF-IDF scores are lower than neural embeddings → 0.30 is a realistic floor.
    # Swap in SentenceTransformer and raise this to 0.75–0.85 for production.
    SIMILARITY_THRESHOLD : float = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

    # ── Phase 2 · Content Engine ──────────────────────────────────────────────
    MAX_POST_CHARS : int = 280          # hard cap enforced in content_engine.py

    # ── Phase 3 · Combat Engine ───────────────────────────────────────────────
    # Number of prior thread messages packed into the RAG context window.
    # Higher = more context, higher token cost.
    MAX_THREAD_CONTEXT : int = 20

    # ─────────────────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Hard-fail at startup if required secrets are missing.
        Called once in main.py before any phase runs.
        """
        errors = []

        if not self.GROQ_API_KEY:
            errors.append(
                "GROQ_API_KEY is missing.\n"
                "    → Get a free key at https://console.groq.com\n"
                "    → Copy .env.example to .env and paste it in."
            )

        if errors:
            raise EnvironmentError(
                "Configuration errors found:\n\n" + "\n\n".join(errors)
            )

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"model={self.LLM_MODEL!r}, "
            f"threshold={self.SIMILARITY_THRESHOLD}, "
            f"max_post={self.MAX_POST_CHARS}, "
            f"max_thread={self.MAX_THREAD_CONTEXT}, "
            f"api_key={'SET' if self.GROQ_API_KEY else 'NOT SET'})"
        )


settings = Settings()
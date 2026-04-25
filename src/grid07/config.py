import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# project paths
ROOT_DIR : Path = Path(__file__).resolve().parents[2]
LOGS_DIR : Path = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE : Path = LOGS_DIR / "execution_log.txt"


# custom file handler — writes logs to execution_log.txt
# "##" messages become section headers, rest are plain lines
class TextFileHandler(logging.FileHandler):

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("##"):
            self.stream.write(f"\n{msg.lstrip('# ')}\n\n")
        else:
            formatted = self.format(record)
            self.stream.write(f"{formatted}\n")
        self.flush()


# console formatter — strips "##" from section headers so terminal looks clean
class _ConsoleSafeFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if record.getMessage().startswith("##"):
            return f"\n{'─' * 60}\n {record.getMessage().lstrip('# ').upper()}\n{'─' * 60}"
        return formatted


# sets up console + text file handlers (runs once at import)
def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if root.handlers:
        return

    for lib in ("httpx", "httpcore", "huggingface_hub",
                "sentence_transformers", "transformers", "filelock"):
        logging.getLogger(lib).setLevel(logging.ERROR)

    # console — all grid07.* and __main__ logs
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        _ConsoleSafeFormatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    # file — only __main__ summaries + "##" phase headers
    file_handler = TextFileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
    file_handler.addFilter(lambda r: r.name.startswith(("grid07", "__main__", "main")) or r.getMessage().startswith("##"))
    root.addHandler(file_handler)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Execution Logs\n\n")


_setup_logging()


# returns a named logger for the given module
# in: name(str) | out: logging.Logger
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Settings — all config values loaded from .env
# GROQ_API_KEY is required; rest have defaults
class Settings:

    # Groq LLM
    GROQ_API_KEY : str   = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   : str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # router similarity cutoff
    SIMILARITY_THRESHOLD : float = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))

    # max characters allowed per generated post
    MAX_POST_CHARS : int = 280

    # how many prior thread messages go into RAG context
    MAX_THREAD_CONTEXT : int = 20

    # in: none | out: raises EnvironmentError if GROQ_API_KEY missing
    def validate(self) -> None:
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
            f"model={self.GROQ_MODEL!r}, "
            f"threshold={self.SIMILARITY_THRESHOLD}, "
            f"max_post={self.MAX_POST_CHARS}, "
            f"max_thread={self.MAX_THREAD_CONTEXT}, "
            f"api_key={'SET' if self.GROQ_API_KEY else 'NOT SET'})"
        )


settings = Settings()
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

ROOT_DIR = Path(__file__).resolve().parents[2]   # grid07/
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "execution_log.md"

class MarkdownFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        if msg.startswith("##"):
            self.stream.write(f"\n{msg}\n")
        else:
            self.stream.write(f"{msg}\n")
        self.flush()

def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if root.handlers:
        return

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(console.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s", datefmt="%H:%M:%S")))
    root.addHandler(console)

    # Markdown file handler
    md = MarkdownFileHandler(LOG_FILE, mode="w", encoding="utf-8")
    md.setLevel(logging.DEBUG)
    md.setFormatter(console.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s", datefmt="%H:%M:%S")))
    root.addHandler(md)

    # Write the markdown template header once
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# Execution Logs\n")


_setup_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Settings:

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
    MAX_POST_CHARS: int = int(os.getenv("MAX_POST_CHARS", 200))
    MAX_THREAD_CONTEXT: int =int(os.getenv("MAX_THREAD_CONTEXT", 10))


    def validate(self) -> None:
        errors = []
        if not self.GROQ_API_KEY:
            errors.append(
                "GROQ_API_KEY is missing.\n"
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
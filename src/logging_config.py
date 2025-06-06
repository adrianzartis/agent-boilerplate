import logging
from pathlib import Path


def setup_logging(log_file: str = "app.log") -> None:
    """Configure root logger to write detailed logs to file."""
    log_path = Path(log_file).resolve()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ],
    )

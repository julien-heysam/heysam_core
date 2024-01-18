import os
import logging
import logging.config
from pathlib import Path

from pythonjsonlogger.jsonlogger import JsonFormatter
from pydantic_settings import BaseSettings
from rich.logging import RichHandler
from dotenv import load_dotenv

load_dotenv(override=True)


class RichCustomFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rich_handler = RichHandler(
            rich_tracebacks=True, 
            tracebacks_suppress=[], 
            tracebacks_show_locals=True
        )

    def format(self, record):
        return super().format(record)


class ProjectApiKeys(BaseSettings):
    PINECONE_ENV: str = os.environ.get("PINECONE_ENV", "")
    PINECONE_INDEX: str = os.environ.get("PINECONE_INDEX", "")
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")

    COMET_API_KEY: str = os.environ.get("COMET_API_KEY", "")
    APIFY_API_TOKEN: str = os.environ.get("APIFY_API_TOKEN", "")
    NOTION_API_KEY: str = os.environ.get("NOTION_API_KEY", "")
    PROMPTLAYER_API_KEY: str = os.environ.get("PROMPTLAYER_API_KEY", "")
    VOYAGE_API_KEY: str = os.environ.get("VOYAGE_API_KEY", "")
    COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")
    
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_ORGANIZATION: str = os.environ.get("OPENAI_ORGANIZATION", "")


class ProjectPaths(BaseSettings):
    ROOT_PATH: Path = Path(__file__).parent.parent

    PROJECT_PATH: Path = ROOT_PATH / "src"
    QUERIES_PATH: Path = PROJECT_PATH / "queries"

    DATA_PATH: Path = ROOT_PATH / "data"
    LOGS_DATA: Path = DATA_PATH / "logs"
    RAW_DATA: Path = DATA_PATH / "raw"
    INTERIM_DATA: Path = DATA_PATH / "interim"
    EXTERNAL_DATA: Path = DATA_PATH / "external"
    PROCESSED_DATA: Path = DATA_PATH / "processed"

    MODEL_DATA: Path = ROOT_PATH / "models"


class ProjectEnvs(BaseSettings):
    ENV: str = os.environ.get("ENV", "production").upper()


PROJECT_ENV = ProjectEnvs()
PROJECT_PATHS = ProjectPaths()
PROJECT_API_KEYS = ProjectApiKeys()
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "console": {
            "()": RichCustomFormatter,
            "format": "%(asctime)s | %(message)s",
            "datefmt": "<%d %b %Y | %H:%M:%S>"
        },
        "json_datadog": {
            "()": JsonFormatter,
            "format": "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] "
                    "[dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s "
                    "dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s] - %(message)s",
            "datefmt": "<%d %b %Y | %H:%M:%S>"
        },
    },
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "level": "INFO",
            "formatter": "console",
            "rich_tracebacks": True,
            "tracebacks_show_locals": True,
        },
        "datadog": {
            "class": "logging.StreamHandler",
            "formatter": "json_datadog",
        },
    },
    "loggers": {
        "": {
            "handlers": ["datadog"] if PROJECT_ENV.ENV == "PROD" else ["console"], 
            "level": "INFO" if PROJECT_ENV.ENV == "PROD" else "INFO", 
            "propagate": True
        },
        "sentence_transformers": {
            "handlers": ["datadog"] if PROJECT_ENV.ENV == "PROD" else ["console"], 
            "level": "INFO" if PROJECT_ENV.ENV == "PROD" else "INFO", 
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["datadog"] if PROJECT_ENV.ENV == "PROD" else ["console"], 
            "level": "INFO" if PROJECT_ENV.ENV == "PROD" else "INFO", 
            "propagate": False
        },
    },
}

logging.captureWarnings(True)
logging.config.dictConfig(LOGGING_CONFIG)

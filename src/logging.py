from __future__ import annotations

import logging
import logging.config
from pathlib import Path


class LoggingSystem:
    _ROOT_PACKAGE = str(Path(__name__).parent.absolute())
    ALLOWED_NAMESPACES = (_ROOT_PACKAGE, "__main__")

    CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s [%(levelname)s] %(name)s "
                    "(%(filename)s:%(lineno)d): %(message)s"
                ),
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": logging.DEBUG,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": logging.ERROR,
        },
    }

    @classmethod
    def global_setup(
        cls: type[LoggingSystem],
        verbose: int,
    ) -> None:
        level: int = cls._get_level(verbose)

        logging.config.dictConfig(cls.CONFIG)

        loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]

        for logger in loggers:
            if any(
                logger.name.startswith(ns) for ns in cls.ALLOWED_NAMESPACES
            ):
                logger.setLevel(level)
            else:
                logger.setLevel(logging.ERROR)

    @staticmethod
    def _get_level(verbose: int) -> int:
        match verbose:
            case 0:
                return logging.ERROR
            case 1:
                return logging.INFO
            case _:
                return logging.DEBUG

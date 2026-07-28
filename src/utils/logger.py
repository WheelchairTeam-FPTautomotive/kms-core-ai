import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name="kms_core_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/core_ai.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

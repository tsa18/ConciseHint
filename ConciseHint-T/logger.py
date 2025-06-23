import logging
import os
from datetime import datetime

def setup_logger(name=None, log_dir='logs', log_filename=None, log_level=logging.INFO):
    """
    setting logger

    Args:
        log_dir (str): log dir
        log_filename (str): log file

    Returns:
        logging.Logger: the logger
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_filename is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"log_{now}.txt"

    log_path = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

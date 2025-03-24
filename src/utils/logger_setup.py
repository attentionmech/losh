import logging


def setup_logger():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger

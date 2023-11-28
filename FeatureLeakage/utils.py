import logging


def create_logger(name: str, level=logger.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formetter = logging.get

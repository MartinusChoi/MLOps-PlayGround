import logging
import sys

def get_logger():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    return logger
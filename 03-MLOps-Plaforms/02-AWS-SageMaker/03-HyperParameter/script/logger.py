import logging

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d - %(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(f'{__name__}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
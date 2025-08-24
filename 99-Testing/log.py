import logging

logger = logging.getLogger('TestLog')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('TestLog_logfile.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.debug("Debugging Message")
logger.info('Info Message')
logger.warning('Warning Message')
logger.error('Error Messgae')
logger.critical("Critical Message")

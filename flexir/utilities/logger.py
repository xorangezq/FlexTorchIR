
import sys
from loguru import logger

logger.remove() # suppress default format
logger.add(sys.stderr, format='<level>{level: <8}</level> | <level>{message}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> |')
# [original official format](https://github.com/Delgan/loguru/issues/109#issuecomment-507814421):
# "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

def ASSERT(condition=False, *logs):
    if not condition:
        logger.error(' '.join(['[ASSERT]'] + list(map(str, logs))))

def TENSOR_PEEK(tensor, howmuch=4):
    return tensor.flatten()[:howmuch]

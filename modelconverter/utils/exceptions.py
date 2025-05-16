import sys
from typing import NoReturn

from loguru import logger


class ModelconverterException(BaseException):
    pass


class S3Exception(ModelconverterException):
    pass


class ONNXException(ModelconverterException):
    pass


class SubprocessException(ModelconverterException):
    pass


def exit_with(exception: BaseException, code: int = 1) -> NoReturn:
    logger.exception(exception)
    sys.exit(code)

from logging import getLogger
from typing import NoReturn


class ModelconverterException(BaseException):
    pass


class S3Exception(ModelconverterException):
    pass


class ONNXException(ModelconverterException):
    pass


class SubprocessException(ModelconverterException):
    pass


def exit_with(exception: BaseException, code: int = 1) -> NoReturn:
    logger = getLogger(__name__)
    logger.exception(exception)
    exit(code)

"""Exception types raised by ModelConverter.

Everything the converter itself considers a failure of the conversion
derives from `ModelconverterException`, which the CLI catches to exit
with status 1, as opposed to the status 2 it uses for an unexpected
error. The module also holds `exit_with`, the helper used where an
error is fatal enough to end the process on the spot.
"""

import sys
from typing import NoReturn

from loguru import logger


class ModelconverterException(BaseException):
    """Base class of the exceptions raised by ModelConverter."""


class S3Exception(ModelconverterException):
    """Raised when an operation on S3 storage fails."""


class ONNXException(ModelconverterException):
    """Raised when an ONNX model cannot be inspected or modified."""


def exit_with(exception: BaseException, code: int = 1) -> NoReturn:
    """Log the exception and terminate the process.

    Args:
        exception: The exception to log, together with its traceback.
        code: The exit status to terminate the process with.

    """
    logger.exception(exception)
    sys.exit(code)

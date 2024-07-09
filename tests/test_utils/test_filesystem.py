import os
import random
import shutil
from pathlib import Path

import pytest

from modelconverter.utils.filesystem_utils import (
    download_from_remote,
    upload_file_to_remote,
)

S3_BASE_URL = "s3://luxonis-test-bucket/modelconverter-test/test_data/test_utils/test_s3_tools/"
S3_FILE_URL = f"{S3_BASE_URL}file.txt"
S3_DIR_URL = f"{S3_BASE_URL}dir/"

LOCAL_ROOT = Path("tests/data/test_utils/test_s3_tools/")
LOCAL_FILE_PATH = LOCAL_ROOT / "file.txt"
LOCAL_DIR_PATH = LOCAL_ROOT / "dir"
LOCAL_FILE_IN_DIR_PATH = LOCAL_DIR_PATH / "file_in_dir.txt"

S3_LOCAL_ROOT = LOCAL_ROOT / "from_s3"
S3_LOCAL_FILE_PATH = S3_LOCAL_ROOT / "file.txt"
S3_LOCAL_DIR_PATH = S3_LOCAL_ROOT / "dir"
S3_LOCAL_FILE_IN_DIR_PATH = S3_LOCAL_DIR_PATH / "file_in_dir.txt"

skip_if_no_remote_credentials = pytest.mark.skipif(
    "AWS_ACCESS_KEY_ID" not in os.environ
    or "AWS_SECRET_ACCESS_KEY" not in os.environ
    or "AWS_S3_ENDPOINT_URL" not in os.environ,
    reason="AWS credentials not set",
)


@pytest.fixture(scope="module", autouse=True)
def setup_remote_tests():
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_DIR_PATH.mkdir(parents=True, exist_ok=True)
    LOCAL_FILE_PATH.write_text(f"test {random.randint(0, 100)}")
    LOCAL_FILE_IN_DIR_PATH.write_text(f"test in dir {random.randint(0, 100)}")

    os.environ.pop("MODELCONVERTER_UNSAFE_CACHE", None)

    yield

    shutil.rmtree(LOCAL_ROOT)


@skip_if_no_remote_credentials
def test_dir():
    upload_file_to_remote(
        LOCAL_FILE_IN_DIR_PATH, f"{S3_DIR_URL}/file_in_dir.txt"
    )
    download_from_remote(S3_DIR_URL, S3_LOCAL_ROOT)
    assert S3_LOCAL_FILE_IN_DIR_PATH.exists()
    assert sorted(path.name for path in S3_LOCAL_DIR_PATH.iterdir()) == sorted(
        path.name for path in LOCAL_DIR_PATH.iterdir()
    )
    for path in LOCAL_DIR_PATH.iterdir():
        assert path.read_text() == (S3_LOCAL_DIR_PATH / path.name).read_text()


@skip_if_no_remote_credentials
def test_file():
    upload_file_to_remote(LOCAL_FILE_PATH, S3_FILE_URL)
    download_from_remote(S3_FILE_URL, S3_LOCAL_ROOT)
    assert S3_LOCAL_FILE_PATH.exists()
    assert S3_LOCAL_FILE_PATH.read_text() == LOCAL_FILE_PATH.read_text()

    with open(S3_LOCAL_FILE_PATH, "w") as f:
        f.write("change")

    assert S3_LOCAL_FILE_PATH.read_text() != LOCAL_FILE_PATH.read_text()
    download_from_remote(S3_FILE_URL, S3_LOCAL_ROOT)
    assert S3_LOCAL_FILE_PATH.read_text() == LOCAL_FILE_PATH.read_text()

"""Docker plumbing behind the ``modelconverter`` commands.

Conversions do not run on the host but inside a per-platform image
(``rvc2``, ``rvc3``, ``rvc4`` or ``hailo``), which carries the vendor
conversion tools. This module finds such an image locally, pulls it or
builds it, describes the conversion container -- its mounts,
environment and resource limits -- as a Docker Compose configuration,
and runs the requested command inside it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import suppress
from functools import cache
from http.client import HTTPMessage
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import psutil
import yaml
from docker.utils import parse_repository_tag
from loguru import logger
from luxonis_ml.typing import Params
from luxonis_ml.utils import environ
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

import docker
from modelconverter import __version__
from modelconverter.utils.constants import (
    CONTAINER_SHARED_DIR,
    HOST_OUTPUT_DIR_ENV_VAR,
    get_cache_dir,
    in_docker,
)
from modelconverter.utils.input_staging import claim_cache
from modelconverter.utils.telemetry import telemetry_environment
from modelconverter.utils.tool_versions import (
    get_default_tool_version,
)

UserNamespaceMode = Literal["rootless", "userns", "rootful", "unknown"]


@cache
def docker_user_namespace_mode() -> UserNamespaceMode:
    """Return how the active Docker daemon maps the container's root
    user onto the host.

    - ``rootless``: the daemon itself runs as the invoking user, so
      container root is already that user.
    - ``userns``: the daemon runs as root with user-namespace remapping,
      so container root is a subordinate uid that cannot even be chowned
      to the host user from inside the container.
    - ``rootful``: container root is host root, so what the container
      writes has to be handed back to the invoking user.
    - ``unknown``: the daemon could not be asked.

    The answer cannot change within a run, so it is cached: this is a
    round-trip to the daemon on a path that would otherwise take it once
    per generated compose config.
    """
    try:
        out = subprocess.check_output(
            [docker_bin(), "info", "-f", "{{json .SecurityOptions}}"],
            text=True,
            stderr=subprocess.DEVNULL,
            # A `DOCKER_HOST` pointing at an unreachable daemon must not hang
            # the conversion.
            timeout=30,
        )
    # `docker_bin` raises RuntimeError when docker is not installed at all.
    except (subprocess.SubprocessError, OSError, RuntimeError):
        return "unknown"
    if "rootless" in out:
        return "rootless"
    if "userns" in out:
        return "userns"
    return "rootful"


def get_docker_client_from_active_context() -> docker.DockerClient:
    """Create a Docker client for the active Docker context.

    The daemon endpoint is read from the context ``docker context show``
    reports, so a non-default context (rootless, remote, Colima, ...) is
    honored instead of the environment defaults. TLS is enabled for a
    ``tcp://`` endpoint unless the context skips the verification.

    Returns:
        A client talking to the active context's daemon.

    """
    ctx_name = subprocess.check_output(
        [docker_bin(), "context", "show"], text=True
    ).strip()

    ctx_info_raw = subprocess.check_output(
        [docker_bin(), "context", "inspect", ctx_name]
    )
    ctx_info = json.loads(ctx_info_raw)[0]

    endpoint = ctx_info["Endpoints"]["docker"]
    host = endpoint.get("Host", None)
    tls_skip = endpoint.get("SkipTLSVerify", False)

    kwargs = {}
    if host:
        kwargs["base_url"] = host
    if host and host.startswith("tcp://") and not tls_skip:
        kwargs["tls"] = True

    return docker.DockerClient(**kwargs)


def rvc4_tag_version(version: str) -> str:
    """Remove the build component from a version string.

    Args:
        version: Version string to strip, e.g. ``2.41.0.251128``.

    Returns:
        The version without its build component, e.g. ``2.41.0``.

    Example:
        >>> rvc4_tag_version("2.41.0.251128")
        '2.41.0'
        >>> rvc4_tag_version("2.41.0")
        '2.41.0'

    """
    parts = version.split(".")
    if len(parts) <= 3:
        return version
    return ".".join(parts[:3])


def generate_compose_config(
    image: str,
    gpu: bool = False,
    memory: int | None = None,
    cpus: float | None = None,
    extra_environment: dict[str, str] | None = None,
) -> str:
    """Generate the Compose configuration of the conversion service.

    The service mounts the modelconverter cache and the host's
    ``./output`` directory, forwards the bucket-storage credentials, and
    -- for a rootful daemon -- the host user's uid and gid, so that the
    container can hand the files it wrote back to the invoking user.

    Args:
        image: Image the service runs. An image whose name ends in
            ``-dev`` additionally mounts the host's sources, tests and
            ``pyproject.toml`` over the ones baked into it.
        gpu: Whether to run the container with the ``nvidia`` runtime.
        memory: Memory limit of the container in bytes. ``None`` sets no
            limit.
        cpus: Number of CPU cores the container may use, possibly
            fractional. ``None`` sets no limit.
        extra_environment: Additional environment variables for the
            container, merged over the defaults.

    Returns:
        The Compose configuration as a YAML document.

    """
    environment = {
        "AWS_ACCESS_KEY_ID": environ.AWS_ACCESS_KEY_ID.get_secret_value()
        if environ.AWS_ACCESS_KEY_ID
        else "",
        "AWS_SECRET_ACCESS_KEY": environ.AWS_SECRET_ACCESS_KEY.get_secret_value()
        if environ.AWS_SECRET_ACCESS_KEY
        else "",
        "AWS_S3_ENDPOINT_URL": environ.AWS_S3_ENDPOINT_URL or "",
        "LUXONISML_BUCKET": environ.LUXONISML_BUCKET or "",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "GOOGLE_APPLICATION_CREDENTIALS": "/run/secrets/gcp-credentials",
        # Forwarded so the in-container test suite can fetch model-zoo
        # archives from HubAI.
        "HUBAI_API_KEY": os.getenv("HUBAI_API_KEY", ""),
    }
    # Pass the host user's identity so the container can chown the outputs and
    # cache back to the invoking user on exit (see docker/*/entrypoint.sh).
    # `getuid`/`getgid` are POSIX-only; on other platforms chowning is neither
    # possible nor necessary. Under a user-namespace daemon it is either
    # unnecessary (rootless: container root already *is* the host user) or
    # impossible (userns remap: the host user is not mapped into the
    # container), and doing it anyway hands the files to an unmapped sub-uid.
    namespace_mode = docker_user_namespace_mode()
    if namespace_mode == "unknown":
        logger.warning(
            "Could not determine the Docker daemon's user-namespace mode; "
            "assuming a rootful daemon. If the outputs come out owned by "
            "another user, unset HOST_UID/HOST_GID and check `docker info`."
        )
    if hasattr(os, "getuid") and namespace_mode in {"rootful", "unknown"}:
        environment["HOST_UID"] = str(os.getuid())
        environment["HOST_GID"] = str(os.getgid())
    if extra_environment:
        environment.update(extra_environment)

    cwd = Path.cwd().absolute()
    host_output_dir = cwd / "output"
    environment[HOST_OUTPUT_DIR_ENV_VAR] = str(host_output_dir)
    volumes = [
        f"{get_cache_dir()}:{CONTAINER_SHARED_DIR}",
        f"{host_output_dir}:/app/output",
    ]
    is_dev = image.endswith("-dev")
    # Mount the test suite (excluded from the image via .dockerignore) so a
    # dev container can run it, e.g.
    # `modelconverter shell <t> --dev -c "pytest -m <t>"`. Only for dev images:
    # a plain conversion has no use for it, and the entrypoint hands the mount
    # back to the invoking user on exit -- which has no business touching an
    # unrelated `tests/` that merely happens to sit in the working directory.
    if is_dev and (cwd / "tests").exists():
        volumes.append(f"{cwd / 'tests'}:/app/tests")
    # Same reasoning: the image carries its own pyproject.toml, and in-container
    # tooling reading /app/pyproject.toml must not pick up whatever Python
    # project the user happens to convert from.
    if is_dev and (cwd / "pyproject.toml").exists():
        volumes.append(f"{cwd / 'pyproject.toml'}:/app/pyproject.toml")
    # The conversion tests convert some of the example configs by their
    # repository-relative path, so those have to be reachable too.
    if (cwd / "configs").exists():
        volumes.append(f"{cwd / 'configs'}:/app/configs")
    # In dev images the package is baked in (`pip install -e .`), so a source
    # change would otherwise need an image rebuild to take effect. Mount the
    # host source over it so edits to modelconverter are live in the container.
    if is_dev and (cwd / "modelconverter").exists():
        volumes.append(f"{cwd / 'modelconverter'}:/app/modelconverter")

    service: Params = {
        "environment": environment,
        "volumes": volumes,
        "secrets": ["gcp-credentials"],
        "image": image,
        "entrypoint": "/app/entrypoint.sh",
    }

    limits = {}
    if memory is not None:
        # Compose reads a bare number as bytes, so the limit is handed over
        # already parsed rather than in whatever spelling the user typed.
        limits["memory"] = str(memory)
    if cpus is not None:
        limits["cpus"] = str(cpus)

    if limits:
        service["deploy"] = {"resources": {"limits": limits}}

    if gpu:
        service["runtime"] = "nvidia"

    config = {
        "services": {"modelconverter": service},
        "secrets": {
            "gcp-credentials": {
                "file": environ.GOOGLE_APPLICATION_CREDENTIALS.get_secret_value()
                if environ.GOOGLE_APPLICATION_CREDENTIALS
                else tempfile.NamedTemporaryFile(delete=False).name,  # noqa: SIM115
            }
        },
    }

    return yaml.dump(config)


def check_docker() -> None:
    """Check that Docker commands can be run from here.

    Raises:
        RuntimeError: If this process already runs inside a container,
            or if Docker is not installed on this system.

    """
    if in_docker():
        raise RuntimeError(
            "Already running in Docker, cannot run Docker commands from within Docker."
        )
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not installed on this system.")


def docker_bin() -> str:
    """Return the path of the ``docker`` executable.

    Returns:
        The path ``docker`` was found at.

    Raises:
        RuntimeError: If Docker is not installed on this system.

    """
    docker_path = shutil.which("docker")
    if docker_path is None:
        raise RuntimeError("Docker is not installed on this system.")
    return docker_path


# NOTE: docker SDK is not used here because it's too slow
def docker_build(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str | None = None,
    image: str | None = None,
) -> str:
    """Build the Docker image of the given platform.

    Args:
        platform: Platform to build the image for.
        bare_tag: Suffix of the image tag, appended to the tool version.
            ``dev`` additionally installs the test tooling into the
            image; for RVC4 any other tag first prepares a clean build
            environment with `prepare_build_environment`.
        version: Version of the underlying conversion tools. Defaults to
            the platform's default version.
        image: Full name of the image to build. If it carries no tag,
            the ``<version>-<bare_tag>`` tag is appended. Defaults to
            ``luxonis/modelconverter-<platform>``.

    Returns:
        The name of the built image, including its tag.

    Raises:
        RuntimeError: If the ``docker build`` invocation fails.

    """
    check_docker()

    if version is None:
        version = get_default_tool_version(platform)

    tag_version = rvc4_tag_version(version) if platform == "rvc4" else version
    if platform == "rvc4" and bare_tag != "dev":
        build_dir = prepare_build_environment(platform, version)
    else:
        build_dir = Path()

    tag = f"{tag_version}-{bare_tag}"

    if image is not None:
        _, image_tag = parse_repository_tag(image)
        if image_tag is None:
            image = f"{image}:{tag}"
    else:
        image = f"luxonis/modelconverter-{platform}:{tag}"

    args = [
        docker_bin(),
        "build",
        "-f",
        str(build_dir / "docker" / platform / "Dockerfile"),
        "-t",
        image,
        "--load",
        str(build_dir),
    ]
    if version is not None:
        args += ["--build-arg", f"VERSION={version}"]
    if bare_tag == "dev":
        # Dev images also carry the test/coverage tooling so the suite can be
        # run inside the container (e.g. `modelconverter shell <t> --dev -c pytest`).
        args += ["--build-arg", "DEV=true"]
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to build the docker image")
    return image


def prepare_build_environment(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"], version: str
) -> Path:
    """Prepare a directory the platform's image can be built from.

    Downloads and extracts the modelconverter sources of the running
    version and puts the SNPE archive of ``version`` in place, either by
    copying a locally available one or by downloading it.

    Args:
        platform: Platform to prepare the build for. Only ``rvc4`` is
            supported.
        version: Version of the conversion tools whose SNPE archive the
            build needs.

    Returns:
        Path to the extracted source tree to build the image from.

    Raises:
        NotImplementedError: If ``platform`` is not ``rvc4``.

    """
    if platform != "rvc4":
        raise NotImplementedError(
            "Fully automatic docker build is only implemented for RVC4"
        )

    build_path = Path(".build", platform)
    build_path.mkdir(parents=True, exist_ok=True)
    _download_file(
        f"https://github.com/luxonis/modelconverter/archive/refs/tags/v{__version__}-beta.zip",
        build_path / f"modelconverter-{__version__}-beta.zip",
        fallback_url="https://github.com/luxonis/modelconverter/archive/refs/heads/main.zip",
    )
    with zipfile.ZipFile(
        build_path / f"modelconverter-{__version__}-beta.zip"
    ) as z:
        z.extractall(build_path)

    if (p := Path("docker", "extra_packages", f"snpe-{version}.zip")).exists():
        shutil.copy(p, build_path / f"modelconverter-{__version__}-beta" / p)

    elif not (build_path / f"modelconverter-{__version__}-beta" / p).exists():
        download_snpe_archive(
            version,
            build_path
            / f"modelconverter-{__version__}-beta"
            / "docker"
            / "extra_packages",
        )

    return build_path / f"modelconverter-{__version__}-beta"


def download_snpe_archive(version: str, dest: Path) -> Path:
    """Download the SNPE archive of the given version.

    Args:
        version: Version of the Qualcomm AI Runtime (SNPE) to download.
        dest: Directory to place the archive in. Created if missing.

    Returns:
        Path to the archive. If it is already present, it is returned
        without downloading anything.

    Raises:
        RuntimeError: If the download fails. The message explains how to
            download the archive manually.

    """
    archive_path = dest / f"snpe-{version}.zip"
    if archive_path.exists():
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{version}/v{version}.zip"
    )
    logger.warning(
        "SNPE archive not found at {}; attempting download from {}",
        archive_path,
        url,
    )
    try:
        _download_file(url, archive_path)
    except (HTTPError, URLError, RuntimeError) as e:
        msg = (
            f"Failed to download SNPE archive from {url}: {e}. "
            "Download it manually from "
            "https://softwarecenter.qualcomm.com/catalog/item/"
            "Qualcomm_AI_Runtime_Community and save it as "
            f"{archive_path}."
        )
        raise RuntimeError(msg) from e

    return archive_path


def _download_file(
    url: str, dest: Path, *, fallback_url: str | None = None
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"Refusing to download from non-HTTPS URL: {url}")

    tmp_path: Path | None = None
    try:
        request = Request(url, headers={"User-Agent": "modelconverter"})  # noqa: S310
        with urlopen(request, timeout=30) as response:  # noqa: S310
            if response.status >= 400:
                raise RuntimeError(
                    f"HTTP {response.status} while downloading {url}"
                )
            headers: HTTPMessage = response.headers
            length = headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            with tempfile.NamedTemporaryFile(
                delete=False, dir=dest.parent, suffix=".zip"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        "Downloading SNPE archive", total=total
                    )
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        progress.update(task, advance=len(chunk))
        tmp_path.replace(dest)
    except Exception as e:
        if fallback_url:
            logger.warning(
                f"Failed to download from {url}: {e}. Attempting fallback URL {fallback_url}..."
            )
            _download_file(fallback_url, dest)
        else:
            raise
    finally:
        if tmp_path is not None and tmp_path.exists() and not dest.exists():
            tmp_path.unlink(missing_ok=True)


# We cannot simply call `docker pull` in a subprocess because
# it interactively asks for login credentials if the image is private.
def pull_image(client: docker.DockerClient, image: str) -> str:
    """Pull an image, showing a progress bar for each of its layers.

    The Docker SDK is used instead of a ``docker pull`` subprocess
    because the latter asks for login credentials interactively when the
    image is private.

    Args:
        client: Docker client to pull with.
        image: Image to pull, optionally including a tag.

    Returns:
        The name of the pulled image.

    """
    repository, tag = parse_repository_tag(image)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        bars = {}
        for log in client.api.pull(repository, tag=tag, stream=True):
            log = json.loads(log)
            status = log["status"]
            if status in {"Downloading", "Extracting"}:
                id = log["id"]
                detail = log["progressDetail"]
                if id not in bars:
                    bars[id] = progress.add_task(
                        f"{id} [{status}]:",
                        completed=detail["current"],
                        total=detail["total"],
                    )
                else:
                    progress.update(
                        bars[id],
                        completed=detail["current"],
                        total=detail["total"],
                        description=f"{id} [{status}]:",
                    )
    return image


def get_docker_image(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str,
    image: str | None = None,
) -> str:
    """Return an image to run the given platform's conversion in.

    A matching local image is used if there is one. Otherwise the
    candidate images are pulled from ``ghcr.io``, and should that fail
    as well, the image is built locally.

    Args:
        platform: Platform the image is for.
        bare_tag: Suffix of the image tag, appended to the tool version,
            e.g. ``latest`` or ``dev``.
        version: Version of the underlying conversion tools.
        image: Full name of the image to use. If it carries no tag, the
            ``<version>-<bare_tag>`` tag is appended. Defaults to
            ``luxonis/modelconverter-<platform>``.

    Returns:
        The name of the image to run, including its tag.

    """
    check_docker()

    local_image = get_local_docker_image(platform, bare_tag, version, image)
    if local_image is not None:
        return local_image

    candidate_images = _get_candidate_docker_images(
        platform, bare_tag, version, image
    )
    return _get_or_build_docker_image(
        platform, bare_tag, version, candidate_images, image
    )


def _get_candidate_docker_images(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str,
    image: str | None = None,
) -> list[str]:
    tag_version = rvc4_tag_version(version) if platform == "rvc4" else version
    tag = f"{tag_version}-{bare_tag}"

    if image is not None:
        image_repo, image_tag = parse_repository_tag(image)
        if image_tag is None:
            image = f"{image_repo}:{tag}"
    else:
        image_repo = f"luxonis/modelconverter-{platform}"
        image_tag = None
        image = f"{image_repo}:{tag}"

    candidate_images = [image]
    # Add full version if the specified RVC4 tag includes a build number
    # (e.g. version=2.32.6.250402 instead of version=2.32.6).
    if platform == "rvc4" and tag_version != version and image_tag is None:
        candidate_images.append(f"{image_repo}:{version}-{bare_tag}")

    return candidate_images


def get_local_docker_image(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str,
    image: str | None = None,
) -> str | None:
    """Return a matching image already present on the local daemon.

    Both the bare candidate names and their ``docker.io`` and
    ``ghcr.io`` spellings are looked for among the local images.

    Args:
        platform: Platform the image is for.
        bare_tag: Suffix of the image tag, appended to the tool version,
            e.g. ``latest`` or ``dev``.
        version: Version of the underlying conversion tools.
        image: Full name of the image to look for. If it carries no tag,
            the ``<version>-<bare_tag>`` tag is appended. Defaults to
            ``luxonis/modelconverter-<platform>``.

    Returns:
        The full name of the first matching local image, including its
        tag, or ``None`` if no candidate is available locally.

    """
    check_docker()

    candidate_images = _get_candidate_docker_images(
        platform, bare_tag, version, image
    )
    client = get_docker_client_from_active_context()
    candidate_tags = set()
    for candidate in candidate_images:
        candidate_tags.add(candidate)
        candidate_tags.add(f"docker.io/{candidate}")
        candidate_tags.add(f"ghcr.io/{candidate}")

    for docker_image in client.images.list():
        tags = candidate_tags & set(docker_image.tags)
        if tags:
            return next(iter(tags))

    return None


def _get_or_build_docker_image(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str,
    candidate_images: list[str],
    image: str | None = None,
) -> str:
    client = get_docker_client_from_active_context()
    for candidate in candidate_images:
        logger.warning(
            f"Image '{candidate}' not found locally, pulling "
            f"the latest image from 'ghcr.io/{candidate}'..."
        )

        with suppress(Exception):
            return pull_image(client, f"ghcr.io/{candidate}")

    logger.error("Failed to pull the image, building it locally...")
    return docker_build(platform, bare_tag, version, image)


def docker_exec(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    *args: str,
    bare_tag: str,
    use_gpu: bool,
    version: str | None = None,
    image: str | None = None,
    memory: int | None = None,
    cpus: float | None = None,
) -> None:
    """Run a command inside the given platform's container.

    Creates the host directories mounted into the container, writes a
    temporary Compose file for it and runs the command with
    ``docker compose run``. The arguments are handed to the container's
    entrypoint as ``argv`` and never re-evaluated by a shell.

    .. note::
        This never returns: the host process exits with the container's
        return code.

    Args:
        platform: Platform whose image the command runs in.
        *args: The command and its arguments.
        bare_tag: Suffix of the image tag, appended to the tool version,
            e.g. ``latest`` or ``dev``.
        use_gpu: Whether to give the container the GPU. Only has an
            effect for the ``hailo`` platform.
        version: Version of the underlying conversion tools. Defaults to
            the platform's default version.
        image: Full name of the image to use. Defaults to the official
            image of the platform.
        memory: Memory limit of the container in bytes. ``None`` sets no
            limit.
        cpus: Number of CPU cores the container may use, possibly
            fractional. ``None`` sets no limit.

    Raises:
        SystemExit: Always, carrying the container's return code.

    """
    version = version or get_default_tool_version(platform)
    image = get_docker_image(platform, bare_tag, version, image)

    # Create the writable host directories up front so they are owned by the
    # invoking user (the cache also holds the staged inputs, and `./output`
    # receives the conversion results).
    get_cache_dir().mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "output").mkdir(parents=True, exist_ok=True)
    # The container downloads remote inputs straight into the cache mount, so
    # `cache clean` in another terminal must see this run even when nothing
    # was staged host-side.
    claim_cache()

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            generate_compose_config(
                image,
                gpu=use_gpu and platform == "hailo",
                memory=memory,
                cpus=cpus,
                extra_environment={
                    **telemetry_environment(),
                    # Lets the in-container test suite auto-select this
                    # platform's conversion tests (see tests/conftest.py).
                    "MODELCONVERTER_PLATFORM": platform,
                    # Lets conversion fixtures select tool-version-specific
                    # test assets.
                    "MODELCONVERTER_TOOL_VERSION": version,
                },
            ).encode()
        )

    # The arguments are passed through as argv, never re-evaluated: the
    # entrypoints run `exec modelconverter "$@"`. They used to build a string
    # and `eval` it, which is what the double quotes here were once rewritten
    # for -- doing so now would only corrupt an inline JSON override.
    sys.exit(
        subprocess.run(
            [
                docker_bin(),
                "compose",
                "-f",
                f.name,
                "run",
                "--rm",
                "--remove-orphans",
                "modelconverter",
                *args,
            ],
            env=os.environ,
            check=False,
        ).returncode
    )


def get_container_memory_limit() -> int:
    """Return the memory limit of the current container in bytes."""
    # cgroup v2 (common on modern Linux/Docker)
    cgroup_v2_path = Path("/sys/fs/cgroup/memory.max")
    if cgroup_v2_path.exists():
        val = cgroup_v2_path.read_text().strip()
        if val.isdigit():
            return int(val)
        if val == "max":
            return psutil.virtual_memory().available
    # fallback to cgroup v1
    cgroup_v1_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_v1_path.exists():
        val = cgroup_v1_path.read_text().strip()
        if val.isdigit():
            return int(val)
    return psutil.virtual_memory().available


def get_container_memory_available() -> int:
    """Return the bytes of memory still available to this container.

    That is its memory limit less the resident memory of every process
    running inside it, never less than zero.
    """
    limit = get_container_memory_limit()
    # sum RSS of all processes in the container
    total_usage = 0
    for p in psutil.process_iter():
        try:
            total_usage += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    available = limit - total_usage
    return max(0, available)

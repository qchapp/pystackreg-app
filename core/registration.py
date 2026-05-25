"""
Pure backend registration functions for pystackreg-app.

These functions implement the three image-registration workflows as
file-based, MCP-friendly operations. They take TIFF file paths as inputs
and return the path to the output TIFF file. No Gradio objects are returned.

Gradio MCP uses function names, type hints, and docstrings to build MCP
tool schemas, so all three are kept clear and complete.
"""

import ipaddress
import os
import socket
import tempfile
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np
import tifffile
from pystackreg import StackReg

from core.utils import WORK_DIR, DEMO_DIR, get_sr_mode, load_stack, normalize_stack

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

VALID_MODES = {"TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"}

# Maximum size allowed for HTTP downloads (prevents resource-exhaustion attacks).
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
_DOWNLOAD_TIMEOUT = 30  # seconds

# Valid TIFF magic byte sequences (little/big-endian classic and BigTIFF).
_TIFF_MAGIC = (
    b"II\x2A\x00",  # little-endian TIFF
    b"MM\x00\x2A",  # big-endian TIFF
    b"II\x2B\x00",  # little-endian BigTIFF
    b"MM\x00\x2B",  # big-endian BigTIFF
)


def _block_private_url(url: str) -> None:
    """Raise ValueError if *url* resolves to a private, loopback, link-local,
    reserved, or multicast address (SSRF protection)."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"Could not parse host from URL: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{host}': {exc}") from exc
    for info in infos:
        addr_str = info[4][0]
        try:
            addr = ipaddress.ip_address(addr_str)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            raise ValueError(
                f"Requests to private or internal addresses are not allowed "
                f"('{host}' resolved to {addr})."
            )


def _download_tiff_to_work_dir(url: str, label: str) -> str:
    """Download *url* to a temp file in WORK_DIR, validating TIFF magic and size.

    Uses chunked streaming so that the full file is never held in memory.
    Raises ValueError on SSRF, size-limit, or magic-byte failures.
    Cleans up the temp file before raising on any error.
    """
    _block_private_url(url)

    fd, local_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)

    total = 0
    first4 = b""

    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp, open(local_path, "wb") as f:  # noqa: S310
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break

                if len(first4) < 4:
                    first4 = (first4 + chunk[: 4 - len(first4)])[:4]
                    if len(first4) == 4 and not any(first4.startswith(magic) for magic in _TIFF_MAGIC):
                        raise ValueError(f"{label} does not appear to be a valid TIFF file.")

                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"{label} exceeds the maximum allowed download size of "
                        f"{_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB."
                    )

                f.write(chunk)

        if total == 0:
            raise ValueError(f"{label} is empty.")

        if len(first4) < 4 or not any(first4.startswith(magic) for magic in _TIFF_MAGIC):
            raise ValueError(f"{label} does not appear to be a valid TIFF file.")

        return local_path

    except Exception:
        try:
            os.unlink(local_path)
        except FileNotFoundError:
            pass
        raise


def _resolve_path(path_or_url: str, label: str = "file") -> str:
    """Return a local, sandbox-safe path for *path_or_url*.

    - If the value starts with ``http://`` or ``https://``, the file is
      downloaded to WORK_DIR and the local path is returned. This is the
      intended flow for MCP clients, which cannot upload files directly.
    - Otherwise the value is treated as a local path and must resolve within
      the app sandbox enforced by _require_file(): WORK_DIR (for outputs from
      previous tool calls) or DEMO_DIR (for cached demo files).
    """
    if path_or_url.startswith(("http://", "https://")):
        return _download_tiff_to_work_dir(path_or_url, label)
    _require_file(path_or_url, label)
    return path_or_url


def _validate_mode(mode: str) -> None:
    """Raise ValueError if *mode* is not a supported transformation mode."""
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid transformation mode '{mode}'. "
            f"Must be one of: {', '.join(sorted(VALID_MODES))}."
        )


def _validate_index(idx: int, stack_len: int, name: str = "frame index") -> None:
    """Raise IndexError if *idx* is outside [0, stack_len)."""
    if not (0 <= idx < stack_len):
        raise IndexError(
            f"{name} {idx} is out of range for a stack with {stack_len} frame(s) "
            f"(valid range: 0 to {stack_len - 1})."
        )


def _require_file(path: str, label: str = "file") -> None:
    """Raise an error if *path* does not exist or is outside the app's sandbox.

    Allowed locations are WORK_DIR (outputs from previous tool calls, enabling
    tool chaining) and DEMO_DIR (cached demo files). Symlinks are resolved
    first to prevent traversal attacks.

    Remote MCP clients should pass HTTP/HTTPS URLs; _resolve_path() downloads
    them to WORK_DIR automatically.
    """
    real = os.path.realpath(path)
    sandboxes = (os.path.realpath(WORK_DIR), os.path.realpath(DEMO_DIR))
    if not any(real == s or real.startswith(s + os.sep) for s in sandboxes):
        raise ValueError(
            f"{label} must be an HTTP/HTTPS URL or a path returned by a previous "
            "tool call. Pass a URL and it will be downloaded automatically."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


# ---------------------------------------------------------------------------
# Private computation helpers (array-in / array-out, no file I/O)
# ---------------------------------------------------------------------------

def _run_align_to_reference(
    stack: np.ndarray, ref_frame: np.ndarray, mode: str
) -> np.ndarray:
    """Register every frame in *stack* against *ref_frame*. Returns normalised uint8 array."""
    sr = StackReg(get_sr_mode(mode))
    return normalize_stack(np.stack([sr.register_transform(ref_frame, fr) for fr in stack]))


def _run_align_to_stack(
    ref_stack: np.ndarray, mov_stack: np.ndarray, mode: str
) -> np.ndarray:
    """Register every frame in *mov_stack* against the first frame of *ref_stack*.
    Returns normalised uint8 array."""
    sr = StackReg(get_sr_mode(mode))
    return normalize_stack(np.stack([sr.register_transform(ref_stack[0], fr) for fr in mov_stack]))


# ---------------------------------------------------------------------------
# Public backend functions (exposed as MCP tools)
# ---------------------------------------------------------------------------


def align_stack_to_reference(
    stack_file: str,
    reference_index: int = 0,
    mode: str = "RIGID_BODY",
    external_reference_file: Optional[str] = None,
    external_reference_index: int = 0,
) -> str:
    """
    Align every frame in a TIFF stack to a chosen reference frame (intra-stack alignment).

    Each frame in *stack_file* is registered to the selected reference frame using
    the chosen transformation model. The reference frame can come from the same
    stack or from a separate external TIFF stack.

    Args:
        stack_file: Path to the input TIFF stack whose frames will be aligned.
        reference_index: Zero-based index of the reference frame inside
            *stack_file* (ignored when *external_reference_file* is provided).
            Default is 0.
        mode: Transformation model for registration. One of: TRANSLATION,
            RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR. Default is RIGID_BODY.
        external_reference_file: Optional path to an external TIFF stack from
            which the reference frame is taken. When provided, *reference_index*
            is ignored and *external_reference_index* is used instead.
        external_reference_index: Zero-based index of the reference frame inside
            *external_reference_file*. Default is 0.

    Returns:
        Path to the aligned output TIFF file (same number of frames as input).

    Raises:
        FileNotFoundError: If *stack_file* or *external_reference_file* does not
            exist on disk.
        IndexError: If *reference_index* or *external_reference_index* is out of
            range for the corresponding stack.
        ValueError: If *mode* is not one of the supported transformation modes.
    """
    _validate_mode(mode)
    stack_file = _resolve_path(stack_file, "stack_file")

    stack = load_stack(stack_file)

    if external_reference_file is not None:
        external_reference_file = _resolve_path(external_reference_file, "external_reference_file")
        ext_stack = load_stack(external_reference_file)
        _validate_index(external_reference_index, len(ext_stack), "external_reference_index")
        ref_frame = ext_stack[external_reference_index]
    else:
        _validate_index(reference_index, len(stack), "reference_index")
        ref_frame = stack[reference_index]

    aligned = _run_align_to_reference(stack, ref_frame, mode)

    fd, out_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(out_path, aligned, photometric="minisblack")
    return out_path


def align_stack_to_stack(
    reference_stack_file: str,
    moving_stack_file: str,
    mode: str = "RIGID_BODY",
) -> str:
    """
    Align every frame in a moving TIFF stack to the first frame of a reference TIFF stack.

    All frames in *moving_stack_file* are registered against the first frame of
    *reference_stack_file* using the specified transformation model.

    Args:
        reference_stack_file: Path to the reference TIFF stack. Its first frame
            is used as the alignment target for all frames in the moving stack.
        moving_stack_file: Path to the moving TIFF stack to align.
        mode: Transformation model for registration. One of: TRANSLATION,
            RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR. Default is RIGID_BODY.

    Returns:
        Path to the aligned output TIFF file (same number of frames as the
        moving stack).

    Raises:
        FileNotFoundError: If *reference_stack_file* or *moving_stack_file* does
            not exist on disk.
        ValueError: If *mode* is not one of the supported transformation modes.
    """
    _validate_mode(mode)
    reference_stack_file = _resolve_path(reference_stack_file, "reference_stack_file")
    moving_stack_file = _resolve_path(moving_stack_file, "moving_stack_file")

    ref_stack = load_stack(reference_stack_file)
    mov_stack = load_stack(moving_stack_file)

    aligned = _run_align_to_stack(ref_stack, mov_stack, mode)

    fd, out_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(out_path, aligned, photometric="minisblack")
    return out_path


def align_frame_to_frame(
    stack_file: str,
    reference_index: int,
    moving_index: int,
    mode: str = "RIGID_BODY",
) -> str:
    """
    Align a single moving frame to a reference frame within the same TIFF stack.

    Registers the frame at *moving_index* against the frame at *reference_index*
    inside *stack_file* and writes the result as a single-frame TIFF.

    Args:
        stack_file: Path to the TIFF stack that contains both frames.
        reference_index: Zero-based index of the reference frame within the stack.
        moving_index: Zero-based index of the frame to align (the moving frame).
        mode: Transformation model for registration. One of: TRANSLATION,
            RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR. Default is RIGID_BODY.

    Returns:
        Path to the aligned output TIFF file (single-frame TIFF).

    Raises:
        FileNotFoundError: If *stack_file* does not exist on disk.
        IndexError: If *reference_index* or *moving_index* is out of range for
            the stack.
        ValueError: If *mode* is not one of the supported transformation modes.
    """
    _validate_mode(mode)
    stack_file = _resolve_path(stack_file, "stack_file")

    stack = load_stack(stack_file)
    _validate_index(reference_index, len(stack), "reference_index")
    _validate_index(moving_index, len(stack), "moving_index")

    sr = StackReg(get_sr_mode(mode))
    aligned = normalize_stack(np.stack([sr.register_transform(stack[reference_index], stack[moving_index])]))[0]

    fd, out_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(out_path, aligned[np.newaxis, ...], photometric="minisblack")
    return out_path

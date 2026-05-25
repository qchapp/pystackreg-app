"""
Pure backend registration functions for pystackreg-app.

These functions implement the three image-registration workflows as
file-based, MCP-friendly operations. They take TIFF file paths as inputs
and return the path to the output TIFF file. No Gradio objects are returned.

Gradio MCP uses function names, type hints, and docstrings to build MCP
tool schemas, so all three are kept clear and complete.
"""

import os
import tempfile
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


def _resolve_path(path_or_url: str, label: str = "file") -> str:
    """Return a local, sandbox-safe path for *path_or_url*.

    - If the value starts with ``http://`` or ``https://``, the file is
      downloaded to WORK_DIR and the local path is returned. This is the
      intended flow for MCP clients, which cannot upload files directly.
    - Otherwise the value is treated as a local path and must resolve within
      the OS temp directory (covers WORK_DIR, DEMO_DIR, and Gradio's upload
      staging area while blocking access to /etc, /home, /var, etc.).
    """
    if path_or_url.startswith(("http://", "https://")):
        fd, local_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
        os.close(fd)
        try:
            urllib.request.urlretrieve(path_or_url, local_path)
        except Exception as exc:
            os.unlink(local_path)
            raise ValueError(f"Failed to download {label} from '{path_or_url}': {exc}") from exc
        return local_path
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
    _validate_index(moving_index, len(stack), "moving_index")

    sr = StackReg(get_sr_mode(mode))
    aligned = normalize_stack(np.stack([sr.register_transform(stack[reference_index], stack[moving_index])]))[0]

    fd, out_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(out_path, aligned[np.newaxis, ...], photometric="minisblack")
    return out_path

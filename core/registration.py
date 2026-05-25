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
from typing import Optional

import numpy as np
import tifffile
from pystackreg import StackReg

from core.utils import WORK_DIR, get_sr_mode, load_stack, normalize_stack

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

VALID_MODES = {"TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"}


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
    """Raise FileNotFoundError if *path* does not point to an existing file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


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
    _require_file(stack_file, "stack_file")

    stack = load_stack(stack_file)

    if external_reference_file is not None:
        _require_file(external_reference_file, "external_reference_file")
        ext_stack = load_stack(external_reference_file)
        _validate_index(external_reference_index, len(ext_stack), "external_reference_index")
        ref_frame = ext_stack[external_reference_index]
    else:
        _validate_index(reference_index, len(stack), "reference_index")
        ref_frame = stack[reference_index]

    sr = StackReg(get_sr_mode(mode))
    aligned = normalize_stack(np.stack([sr.register_transform(ref_frame, fr) for fr in stack]))

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=WORK_DIR).name
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
    _require_file(reference_stack_file, "reference_stack_file")
    _require_file(moving_stack_file, "moving_stack_file")

    ref_stack = load_stack(reference_stack_file)
    mov_stack = load_stack(moving_stack_file)

    sr = StackReg(get_sr_mode(mode))
    aligned = normalize_stack(
        np.stack([sr.register_transform(ref_stack[0], fr) for fr in mov_stack])
    )

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=WORK_DIR).name
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
    _require_file(stack_file, "stack_file")

    stack = load_stack(stack_file)
    _validate_index(reference_index, len(stack), "reference_index")
    _validate_index(moving_index, len(stack), "moving_index")

    sr = StackReg(get_sr_mode(mode))
    aligned = normalize_stack(np.stack([sr.register_transform(stack[reference_index], stack[moving_index])]))[0]

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=WORK_DIR).name
    tifffile.imwrite(out_path, aligned[np.newaxis, ...], photometric="minisblack")
    return out_path

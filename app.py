import gradio as gr
from PIL import Image
import tifffile
import tempfile
import os
from typing import Optional

from core.utils import (
    WORK_DIR, DEMO_DIR, upscale, load_stack,
    _start_cleaner, citation_markdown, documentation_markdown
)
from core.registration import (
    align_stack_to_reference,
    align_stack_to_stack,
    align_frame_to_frame,
    _run_align_to_reference,
    _run_align_to_stack,
    _resolve_path,
)

_start_cleaner()

def _stage_for_backend(src: str) -> str:
    """Return a sandbox-safe path for *src*.

    Files already in WORK_DIR or DEMO_DIR are returned as-is. Files that
    Gradio placed in its own upload staging area are hard-linked into WORK_DIR
    (zero-copy on the same filesystem) so the backend sandbox accepts them.
    Falls back to a regular copy if the hard link fails (cross-device).
    """
    import shutil
    real = os.path.realpath(src)
    sandboxes = (os.path.realpath(WORK_DIR), os.path.realpath(DEMO_DIR))
    if any(real == s or real.startswith(s + os.sep) for s in sandboxes):
        return src
    suffix = os.path.splitext(src)[-1] or ".tif"
    fd, dst = tempfile.mkstemp(suffix=suffix, dir=WORK_DIR)
    os.close(fd)
    os.unlink(dst)  # remove placeholder so os.link can create the entry
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst

def reset_intra_stack():
    return [None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0),
            None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None,
            None, None]

def reset_reference_based():
    return [None, None, None, gr.update(value=0, minimum=0, maximum=0),
            None, gr.update(value=0, minimum=0, maximum=0), None,
            None, None]

def reset_frame_to_frame():
    return [None, gr.update(value=0, minimum=0, maximum=0),
            gr.update(value=0, minimum=0, maximum=0), None, None]

# Registration logic — UI wrappers that call the pure backend functions
def intra_stack_align(f, ref_idx, ext_file, ext_idx, mode):
    if not f:
        raise gr.Error("Please upload a TIFF stack before running alignment.")
    f = _stage_for_backend(f)
    # Load and normalise once — reused for both the UI preview and registration
    orig_stack = load_stack(f)
    fd, orig_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(orig_path, orig_stack, photometric="minisblack")
    n_orig = len(orig_stack)

    if ext_file:
        ref_frame = load_stack(_stage_for_backend(ext_file))[int(ext_idx)]
    else:
        ref_frame = orig_stack[int(ref_idx)]
    aligned = _run_align_to_reference(orig_stack, ref_frame, mode)

    fd, path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(path, aligned, photometric="minisblack")
    return (
        Image.fromarray(orig_stack[0]), gr.update(value=0, maximum=n_orig - 1),
        upscale(Image.fromarray(aligned[0])), gr.update(value=0, maximum=len(aligned) - 1), path,
        orig_path, path,
    )

def reference_align(ref_file, mov_file, mode):
    if not ref_file:
        raise gr.Error("Please upload a reference stack.")
    if not mov_file:
        raise gr.Error("Please upload a moving stack.")
    ref_file = _stage_for_backend(ref_file)
    mov_file = _stage_for_backend(mov_file)
    # Load both stacks once — reference reused for preview and registration
    ref_stack = load_stack(ref_file)
    mov_stack = load_stack(mov_file)
    fd, ref_path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(ref_path, ref_stack, photometric="minisblack")
    n_ref = len(ref_stack)

    aligned = _run_align_to_stack(ref_stack, mov_stack, mode)

    fd, path = tempfile.mkstemp(suffix=".tif", dir=WORK_DIR)
    os.close(fd)
    tifffile.imwrite(path, aligned, photometric="minisblack")
    return (
        Image.fromarray(ref_stack[0]), gr.update(value=0, maximum=n_ref - 1),
        upscale(Image.fromarray(aligned[0])), gr.update(value=0, maximum=len(aligned) - 1), path,
        ref_path, path,
    )

def frame_to_frame_align(file, ref_idx, mov_idx, mode):
    if not file:
        raise gr.Error("Please upload a TIFF stack before running alignment.")
    file = _stage_for_backend(file)
    # Delegate to pure backend
    path = align_frame_to_frame(
        stack_file=file,
        reference_index=int(ref_idx),
        moving_index=int(mov_idx),
        mode=mode,
    )

    # Load aligned frame for UI preview (already normalised — read raw to avoid double-normalisation)
    result_stack = tifffile.imread(path)
    return Image.fromarray(result_stack[0]), path

def _read_frame(path, idx, scale=False):
    """Read a single frame from a TIFF file by index, return a PIL Image."""
    if not path:
        return None
    try:
        frame = tifffile.imread(path, key=int(idx))
        img = Image.fromarray(frame)
        return upscale(img) if scale else img
    except Exception:
        return None

def _count_frames(path: str) -> int:
    """Return the number of frames in a TIFF file without loading pixel data."""
    if not path or not os.path.exists(path):
        return 0
    try:
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages)
    except Exception as exc:
        raise gr.Error("Unable to read the uploaded TIFF file. Please upload a valid, non-corrupt TIFF stack.") from exc

# Interface
with gr.Blocks() as demo:
    # Per-session state — one independent copy per connected user.
    # Avoids the shared-globals race condition where concurrent users
    # would overwrite each other's frame lists and see wrong previews.
    original_path_state = gr.State(None)
    aligned_path_state = gr.State(None)
    ref_path_state = gr.State(None)
    reg_path_state = gr.State(None)

    gr.Markdown("# 🧠 Pystackreg Web Application")
    gr.Markdown(citation_markdown)

    with gr.Accordion("📘 How to Use This App (Click to Expand)", open=False):
        gr.Markdown(documentation_markdown)

    with gr.Tab("📚 Reference-Based Alignment"):
        with gr.Row():
            file_input = gr.File(label="Upload Stack to Register")
            use_ext_ref = gr.Checkbox(label="Use external reference stack")

        gr.Examples(
            examples=[
                ["https://github.com/glichtner/pystackreg/raw/master/examples/data/pc12-unreg.tif"]
            ],
            inputs=[file_input],
            label="Try with Example TIFF"
        )

        with gr.Row():
            reference_frame_slider = gr.Slider(label="Reference Frame (from uploaded stack)", minimum=0, maximum=0, value=0, step=1, visible=True)
            ext_ref_file = gr.File(label="Upload External Reference Stack (.tif)", visible=False)
            ext_ref_slider = gr.Slider(label="Reference Frame (from external stack)", minimum=0, maximum=0, value=0, step=1, visible=False)

        use_ext_ref.change(
            lambda v: (
                gr.update(visible=not v),
                gr.update(visible=v),
                gr.update(visible=v)
            ),
            use_ext_ref,
            [reference_frame_slider, ext_ref_file, ext_ref_slider],
            show_api=False,
        )

        ext_ref_file.change(
            lambda f: gr.update(value=0, maximum=_count_frames(f) - 1) if f else gr.update(value=0, maximum=0),
            ext_ref_file,
            ext_ref_slider,
            show_api=False,
        )

        with gr.Row():
            show_adv = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                        value="RIGID_BODY", visible=False, label="Transformation Mode")

        show_adv.change(lambda v: gr.update(visible=v), show_adv, mode_dropdown, show_api=False)
        run_btn = gr.Button("▶️ Align Stack")

        with gr.Row():
            original_image = gr.Image(label="Original Frame")
            aligned_image = gr.Image(label="Aligned Frame")

        with gr.Row():
            original_slider = gr.Slider(label="Browse Original Stack", minimum=0, maximum=0, value=0, step=1)
            aligned_slider = gr.Slider(label="Browse Aligned Stack", minimum=0, maximum=0, value=0, step=1)

        download = gr.File(label="Download")

        file_input.change(
            lambda f: gr.update(value=0, maximum=_count_frames(f) - 1) if f else gr.update(value=0, maximum=0),
            file_input,
            reference_frame_slider,
            show_api=False,
        )

        run_btn.click(
            intra_stack_align,
            [file_input, reference_frame_slider, ext_ref_file, ext_ref_slider, mode_dropdown],
            [original_image, original_slider, aligned_image, aligned_slider, download,
             original_path_state, aligned_path_state],
            show_api=False,
        )

        original_slider.change(
            lambda i, path: _read_frame(path, i, scale=False),
            [original_slider, original_path_state], original_image, show_api=False,
        )
        aligned_slider.change(
            lambda i, path: _read_frame(path, i, scale=True),
            [aligned_slider, aligned_path_state], aligned_image, show_api=False,
        )

        gr.Button("🔄 Reset Tab").click(
            reset_intra_stack,
            outputs=[
                file_input, reference_frame_slider, ext_ref_file, ext_ref_slider,
                original_image, original_slider, aligned_image, aligned_slider, download,
                original_path_state, aligned_path_state,
            ],
            show_api=False,
        )

    with gr.Tab("🎯 Stack-Based Alignment"):
        with gr.Row():
            ref_input = gr.File(label="Reference Stack")
            mov_input = gr.File(label="Moving Stack")

        gr.Examples(
            examples=[
                [
                    "https://github.com/glichtner/pystackreg/raw/master/examples/data/pc12-unreg.tif",
                    "https://github.com/glichtner/pystackreg/raw/master/examples/data/pc12-reg-translation.tif"
                ]
            ],
            inputs=[ref_input, mov_input],
            label="Try with Example Stacks"
        )

        with gr.Row():
            show_adv_ref = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown_ref = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                            value="RIGID_BODY", visible=False, label="Transformation Mode")

        show_adv_ref.change(lambda v: gr.update(visible=v), show_adv_ref, mode_dropdown_ref, show_api=False)
        ref_btn = gr.Button("▶️ Register")

        with gr.Row():
            ref_image = gr.Image(label="Reference Frame")
            reg_image = gr.Image(label="Registered Frame")

        with gr.Row():
            stack_ref_browse_slider = gr.Slider(label="Browse Ref", minimum=0, maximum=0, value=0, step=1)
            reg_slider = gr.Slider(label="Browse Reg", minimum=0, maximum=0, value=0, step=1)

        download_ref = gr.File(label="Download")

        ref_btn.click(
            reference_align,
            [ref_input, mov_input, mode_dropdown_ref],
            [ref_image, stack_ref_browse_slider, reg_image, reg_slider, download_ref,
             ref_path_state, reg_path_state],
            show_api=False,
        )
        stack_ref_browse_slider.change(
            lambda i, path: _read_frame(path, i, scale=False),
            [stack_ref_browse_slider, ref_path_state], ref_image, show_api=False,
        )
        reg_slider.change(
            lambda i, path: _read_frame(path, i, scale=True),
            [reg_slider, reg_path_state], reg_image, show_api=False,
        )

        gr.Button("🔄 Reset Tab").click(
            reset_reference_based,
            outputs=[ref_input, mov_input, ref_image, stack_ref_browse_slider, reg_image, reg_slider, download_ref,
                     ref_path_state, reg_path_state],
            show_api=False,
        )

    with gr.Tab("🧩 Frame-to-Frame Alignment"):
        with gr.Row():
            frame_file = gr.File(label="Upload Stack")
            ref_idx = gr.Slider(label="Reference Frame", minimum=0, maximum=0, value=0, step=1)
            mov_idx = gr.Slider(label="Moving Frame", minimum=0, maximum=0, value=0, step=1)

        gr.Examples(
            examples=[
                ["https://github.com/glichtner/pystackreg/raw/master/examples/data/pc12-unreg.tif"]
            ],
            inputs=[frame_file],
            label="Try with Example Stack"
        )


        with gr.Row():
            show_adv_ftf = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown_ftf = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                            value="RIGID_BODY", visible=False, label="Transformation Mode")

        show_adv_ftf.change(lambda v: gr.update(visible=v), show_adv_ftf, mode_dropdown_ftf, show_api=False)
        frame_btn = gr.Button("▶️ Register Frame")
        frame_output = gr.Image(label="Registered Output")
        download_ftf = gr.File(label="Download")

        frame_file.change(
            lambda f: [gr.update(value=0, maximum=_count_frames(f) - 1)] * 2 if f else [gr.update(value=0, maximum=0)] * 2,
            frame_file, [ref_idx, mov_idx],
            show_api=False,
        )

        frame_btn.click(
            frame_to_frame_align,
            [frame_file, ref_idx, mov_idx, mode_dropdown_ftf],
            [frame_output, download_ftf],
            show_api=False,
        )

        gr.Button("🔄 Reset Tab").click(
            reset_frame_to_frame,
            outputs=[frame_file, ref_idx, mov_idx, frame_output, download_ftf],
            show_api=False,
        )

    # ---------------------------------------------------------------------------
    # MCP / API-only endpoints — thin wrappers that return the output file 
    # as a Grado FileData so Gradio/MCP can serve it correctly.
    # ---------------------------------------------------------------------------
    def _as_mcp_file(path: str) -> gr.FileData:
        """Wrap a generated local file so Gradio exposes it as a served file."""
        return gr.FileData(
            path=path,
            orig_name=os.path.basename(path),
            mime_type="image/tiff",
            size=os.path.getsize(path),
        )

    def _mcp_align_stack_to_reference(
        stack_file: str,
        reference_index: int = 0,
        mode: str = "RIGID_BODY",
        external_reference_file: Optional[str] = None,
        external_reference_index: int = 0,
    ) -> str:
        """Align every frame in a TIFF stack to a chosen reference frame.

        Each frame in stack_file is registered to the selected reference frame
        using the chosen transformation model. The reference frame can come from
        the same stack or from a separate external TIFF stack.

        Args:
            stack_file: Path or HTTP/HTTPS URL to the input TIFF stack.
            reference_index: Zero-based index of the reference frame inside
                stack_file (ignored when external_reference_file is provided).
                Default is 0.
            mode: Transformation model. One of: TRANSLATION, RIGID_BODY,
                SCALED_ROTATION, AFFINE, BILINEAR. Default is RIGID_BODY.
            external_reference_file: Optional path or URL to an external TIFF
                stack from which the reference frame is taken.
            external_reference_index: Zero-based index of the reference frame
                inside external_reference_file. Default is 0.

        Returns:
            The aligned output TIFF file.
        """
        out = align_stack_to_reference(
            stack_file, reference_index, mode,
            external_reference_file, external_reference_index,
        )
        return _as_mcp_file(out)

    def _mcp_align_stack_to_stack(
        reference_stack_file: str,
        moving_stack_file: str,
        mode: str = "RIGID_BODY",
    ) -> str:
        """Align every frame in a moving TIFF stack to the first frame of a reference stack.

        Args:
            reference_stack_file: Path or HTTP/HTTPS URL to the reference TIFF
                stack. Its first frame is used as the alignment target.
            moving_stack_file: Path or HTTP/HTTPS URL to the moving TIFF stack
                to align.
            mode: Transformation model. One of: TRANSLATION, RIGID_BODY,
                SCALED_ROTATION, AFFINE, BILINEAR. Default is RIGID_BODY.

        Returns:
            The aligned output TIFF file.
        """
        out = align_stack_to_stack(reference_stack_file, moving_stack_file, mode)
        return _as_mcp_file(out)

    def _mcp_align_frame_to_frame(
        stack_file: str,
        reference_index: int,
        moving_index: int,
        mode: str = "RIGID_BODY",
    ) -> str:
        """Align a single moving frame to a reference frame within the same TIFF stack.

        Args:
            stack_file: Path or HTTP/HTTPS URL to the TIFF stack containing both
                frames.
            reference_index: Zero-based index of the reference frame.
            moving_index: Zero-based index of the frame to align.
            mode: Transformation model. One of: TRANSLATION, RIGID_BODY,
                SCALED_ROTATION, AFFINE, BILINEAR. Default is RIGID_BODY.

        Returns:
            The aligned single-frame output TIFF file.
        """
        out = align_frame_to_frame(stack_file, reference_index, moving_index, mode)
        return _as_mcp_file(out)

    gr.api(fn=_mcp_align_stack_to_reference, api_name="align_stack_to_reference")
    gr.api(fn=_mcp_align_stack_to_stack, api_name="align_stack_to_stack")
    gr.api(fn=_mcp_align_frame_to_frame, api_name="align_frame_to_frame")

    # ---------------------------------------------------------------------------
    # Page-load handler (UI only — not an MCP tool)
    # ---------------------------------------------------------------------------
    def load_from_query(request: gr.Request):
        params = request.query_params
        results = [None] * 7  # 7 outputs

        # One-stack file case (for ref-based + frame-to-frame)
        if "file_url" in params:
            try:
                tmp_path = _resolve_path(params["file_url"], "file_url")
                with tifffile.TiffFile(tmp_path) as tf:
                    max_frame = len(tf.pages) - 1

                results[0] = tmp_path  # file_input
                results[1] = gr.update(value=0, maximum=max_frame)  # reference_frame_slider
                results[2] = tmp_path  # frame_file
                results[3] = gr.update(value=0, maximum=max_frame)  # ref_idx
                results[4] = gr.update(value=1 if max_frame >= 1 else 0, maximum=max_frame)  # mov_idx

            except Exception as e:
                print(f"[Error loading file_url] {e}")

        # Two-stack file case (for stack-based alignment)
        if "file_url_1" in params and "file_url_2" in params:
            try:
                results[5] = _resolve_path(params["file_url_1"], "file_url_1")  # ref_input
                results[6] = _resolve_path(params["file_url_2"], "file_url_2")  # mov_input
            except Exception as e:
                print(f"[Error loading file_url_1 or file_url_2] {e}")

        return results

    demo.load(
        load_from_query,
        outputs=[file_input, reference_frame_slider, frame_file, ref_idx, mov_idx, ref_input, mov_input],
        show_api=False,
    )


if __name__ == "__main__":
    demo.launch(mcp_server=True)
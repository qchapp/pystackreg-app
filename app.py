import gradio as gr
import numpy as np
from PIL import Image
import tifffile
import tempfile
import urllib.request
import os

from core.utils import (
    WORK_DIR, normalize_stack, upscale, load_stack,
    _is_demo_url, _demo_path_for_url,
    _start_cleaner, citation_markdown, documentation_markdown
)
from core.registration import (
    align_stack_to_reference,
    align_stack_to_stack,
    align_frame_to_frame,
)

_start_cleaner()

# Reset functions
def reset_intra_stack():
    return [None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0),
            None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None,
            [], []]

def reset_reference_based():
    return [None, None, None, gr.update(value=0, minimum=0, maximum=0),
            None, gr.update(value=0, minimum=0, maximum=0), None,
            [], []]

def reset_frame_to_frame():
    return [None, gr.update(value=0, minimum=0, maximum=0),
            gr.update(value=0, minimum=0, maximum=0), None, None]

# Registration logic — UI wrappers that call the pure backend functions
def intra_stack_align(f, ref_idx, ext_file, ext_idx, mode):
    if not f:
        raise gr.Error("Please upload a TIFF stack before running alignment.")
    # Load original for UI preview
    orig_stack = load_stack(f)
    original_frames = [Image.fromarray(fr) for fr in orig_stack]

    # Delegate to pure backend
    path = align_stack_to_reference(
        stack_file=f,
        reference_index=int(ref_idx),
        mode=mode,
        external_reference_file=ext_file if ext_file else None,
        external_reference_index=int(ext_idx),
    )

    # Load aligned result for UI preview
    aligned_stack = load_stack(path)
    aligned_frames = [upscale(Image.fromarray(fr)) for fr in aligned_stack]

    return (
        original_frames[0], gr.update(value=0, maximum=len(original_frames)-1),
        aligned_frames[0], gr.update(value=0, maximum=len(aligned_frames)-1), path,
        original_frames, aligned_frames,
    )

def reference_align(ref_file, mov_file, mode):
    if not ref_file:
        raise gr.Error("Please upload a reference stack.")
    if not mov_file:
        raise gr.Error("Please upload a moving stack.")
    ref_stack = load_stack(ref_file)
    ref_frames = [Image.fromarray(f) for f in ref_stack]

    # Delegate to pure backend
    path = align_stack_to_stack(
        reference_stack_file=ref_file,
        moving_stack_file=mov_file,
        mode=mode,
    )

    # Load registered result for UI preview
    reg_stack = load_stack(path)
    reg_frames = [upscale(Image.fromarray(f)) for f in reg_stack]

    return (
        ref_frames[0], gr.update(value=0, maximum=len(ref_frames)-1),
        reg_frames[0], gr.update(value=0, maximum=len(reg_frames)-1), path,
        ref_frames, reg_frames,
    )

def frame_to_frame_align(file, ref_idx, mov_idx, mode):
    if not file:
        raise gr.Error("Please upload a TIFF stack before running alignment.")
    # Delegate to pure backend
    path = align_frame_to_frame(
        stack_file=file,
        reference_index=int(ref_idx),
        moving_index=int(mov_idx),
        mode=mode,
    )

    # Load aligned frame for UI preview
    result_stack = load_stack(path)
    return Image.fromarray(result_stack[0]), path

def _count_frames(path: str) -> int:
    """Return the number of frames in a TIFF file without loading pixel data."""
    with tifffile.TiffFile(path) as tf:
        return len(tf.pages)

# Interface
with gr.Blocks() as demo:
    # Per-session state — one independent copy per connected user.
    # Avoids the shared-globals race condition where concurrent users
    # would overwrite each other's frame lists and see wrong previews.
    original_frames_state = gr.State([])
    aligned_frames_state = gr.State([])
    ref_frames_state = gr.State([])
    reg_frames_state = gr.State([])

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
            ref_slider = gr.Slider(label="Reference Frame (from uploaded stack)", minimum=0, maximum=0, value=0, step=1, visible=True)
            ext_ref_file = gr.File(label="Upload External Reference Stack (.tif)", visible=False)
            ext_ref_slider = gr.Slider(label="Reference Frame (from external stack)", minimum=0, maximum=0, value=0, step=1, visible=False)

        use_ext_ref.change(
            lambda v: (
                gr.update(visible=not v),
                gr.update(visible=v),
                gr.update(visible=v)
            ),
            use_ext_ref,
            [ref_slider, ext_ref_file, ext_ref_slider],
            api_name=False,
        )

        ext_ref_file.change(
            lambda f: gr.update(value=0, maximum=_count_frames(f) - 1) if f else gr.update(value=0, maximum=0),
            ext_ref_file,
            ext_ref_slider,
            api_name=False,
        )

        with gr.Row():
            show_adv = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                        value="RIGID_BODY", visible=False, label="Transformation Mode")

        show_adv.change(lambda v: gr.update(visible=v), show_adv, mode_dropdown, api_name=False)
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
            ref_slider,
            api_name=False,
        )

        run_btn.click(
            intra_stack_align,
            [file_input, ref_slider, ext_ref_file, ext_ref_slider, mode_dropdown],
            [original_image, original_slider, aligned_image, aligned_slider, download,
             original_frames_state, aligned_frames_state],
            api_name=False,
        )

        original_slider.change(
            lambda i, frames: frames[i] if 0 <= i < len(frames) else None,
            [original_slider, original_frames_state], original_image, api_name=False,
        )
        aligned_slider.change(
            lambda i, frames: frames[i] if 0 <= i < len(frames) else None,
            [aligned_slider, aligned_frames_state], aligned_image, api_name=False,
        )

        gr.Button("🔄 Reset Tab").click(
            reset_intra_stack,
            outputs=[
                file_input, ref_slider, ext_ref_file, ext_ref_slider,
                original_image, original_slider, aligned_image, aligned_slider, download,
                original_frames_state, aligned_frames_state,
            ],
            api_name=False,
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

        show_adv_ref.change(lambda v: gr.update(visible=v), show_adv_ref, mode_dropdown_ref, api_name=False)
        ref_btn = gr.Button("▶️ Register")

        with gr.Row():
            ref_image = gr.Image(label="Reference Frame")
            reg_image = gr.Image(label="Registered Frame")

        with gr.Row():
            ref_slider = gr.Slider(label="Browse Ref", minimum=0, maximum=0, value=0, step=1)
            reg_slider = gr.Slider(label="Browse Reg", minimum=0, maximum=0, value=0, step=1)

        download_ref = gr.File(label="Download")

        ref_btn.click(
            reference_align,
            [ref_input, mov_input, mode_dropdown_ref],
            [ref_image, ref_slider, reg_image, reg_slider, download_ref,
             ref_frames_state, reg_frames_state],
            api_name=False,
        )
        ref_slider.change(
            lambda i, frames: frames[i] if 0 <= i < len(frames) else None,
            [ref_slider, ref_frames_state], ref_image, api_name=False,
        )
        reg_slider.change(
            lambda i, frames: frames[i] if 0 <= i < len(frames) else None,
            [reg_slider, reg_frames_state], reg_image, api_name=False,
        )

        gr.Button("🔄 Reset Tab").click(
            reset_reference_based,
            outputs=[ref_input, mov_input, ref_image, ref_slider, reg_image, reg_slider, download_ref,
                     ref_frames_state, reg_frames_state],
            api_name=False,
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

        show_adv_ftf.change(lambda v: gr.update(visible=v), show_adv_ftf, mode_dropdown_ftf, api_name=False)
        frame_btn = gr.Button("▶️ Register Frame")
        frame_output = gr.Image(label="Registered Output")
        download_ftf = gr.File(label="Download")

        frame_file.change(
            lambda f: [gr.update(value=0, maximum=_count_frames(f) - 1)] * 2 if f else [gr.update(value=0, maximum=0)] * 2,
            frame_file, [ref_idx, mov_idx],
            api_name=False,
        )

        frame_btn.click(
            frame_to_frame_align,
            [frame_file, ref_idx, mov_idx, mode_dropdown_ftf],
            [frame_output, download_ftf],
            api_name=False,
        )

        gr.Button("🔄 Reset Tab").click(
            reset_frame_to_frame,
            outputs=[frame_file, ref_idx, mov_idx, frame_output, download_ftf],
            api_name=False,
        )

    # ---------------------------------------------------------------------------
    # MCP / API-only endpoints — exposed as MCP tools when mcp_server=True
    # ---------------------------------------------------------------------------
    gr.api(
        fn=align_stack_to_reference,
        api_name="align_stack_to_reference",
    )
    gr.api(
        fn=align_stack_to_stack,
        api_name="align_stack_to_stack",
    )
    gr.api(
        fn=align_frame_to_frame,
        api_name="align_frame_to_frame",
    )

    # ---------------------------------------------------------------------------
    # Page-load handler (UI only — not an MCP tool)
    # ---------------------------------------------------------------------------
    def load_from_query(request: gr.Request):
        params = request.query_params
        results = [None] * 7  # 7 outputs

        # One-stack file case (for ref-based + frame-to-frame)
        if "file_url" in params:
            try:
                url = params["file_url"]
                if _is_demo_url(url):
                    tmp_path = _demo_path_for_url(url)
                    if not os.path.exists(tmp_path):
                        urllib.request.urlretrieve(url, tmp_path)
                else:
                    tmp_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=WORK_DIR).name
                    urllib.request.urlretrieve(url, tmp_path)
                with tifffile.TiffFile(tmp_path) as tf:
                    max_frame = len(tf.pages) - 1

                results[0] = tmp_path  # file_input
                results[1] = gr.update(value=0, maximum=max_frame)  # ref_slider
                results[2] = tmp_path  # frame_file
                results[3] = gr.update(value=0, maximum=max_frame)  # ref_idx
                results[4] = gr.update(value=1 if max_frame >= 1 else 0, maximum=max_frame)  # mov_idx

            except Exception as e:
                print(f"[Error loading file_url] {e}")

        # Two-stack file case (for stack-based alignment)
        if "file_url_1" in params and "file_url_2" in params:
            try:
                u1, u2 = params["file_url_1"], params["file_url_2"]

                if _is_demo_url(u1):
                    tmp_path_1 = _demo_path_for_url(u1)
                    if not os.path.exists(tmp_path_1):
                        urllib.request.urlretrieve(u1, tmp_path_1)
                else:
                    tmp_path_1 = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=WORK_DIR).name
                    urllib.request.urlretrieve(u1, tmp_path_1)

                if _is_demo_url(u2):
                    tmp_path_2 = _demo_path_for_url(u2)
                    if not os.path.exists(tmp_path_2):
                        urllib.request.urlretrieve(u2, tmp_path_2)
                else:
                    tmp_path_2 = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=WORK_DIR).name
                    urllib.request.urlretrieve(u2, tmp_path_2)

                results[5] = tmp_path_1  # ref_input
                results[6] = tmp_path_2  # mov_input
            except Exception as e:
                print(f"[Error loading file_url_1 or file_url_2] {e}")

        return results

    demo.load(
        load_from_query,
        outputs=[file_input, ref_slider, frame_file, ref_idx, mov_idx, ref_input, mov_input],
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch(mcp_server=True)
import gradio as gr
import numpy as np
from PIL import Image
from pystackreg import StackReg
import imageio.v2 as iio
import tifffile
import tempfile

from core.utils import *

# Globals
original_frames, aligned_frames = [], []
ref_frames, mov_frames, reg_frames = [], [], []
custom_stack, reg_result = [], None
full_stack, full_aligned = [], []

# reset
def reset_intra_stack():
    global original_frames, aligned_frames
    original_frames, aligned_frames = [], []
    return [None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None]

def reset_reference_based():
    global ref_frames, mov_frames, reg_frames
    ref_frames, mov_frames, reg_frames = [], [], []
    return [None, None, None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None]

def reset_frame_to_frame():
    global custom_stack, reg_result
    custom_stack, reg_result = [], None
    return [None, gr.update(value=0, minimum=0, maximum=0), gr.update(value=0, minimum=0, maximum=0), None, None]


# Registration Logic
def intra_stack_align(f, ref_idx, ext_file, ext_idx, mode):
    global original_frames, aligned_frames
    stack = load_stack(f)
    original_frames = [Image.fromarray(fr) for fr in stack]
    sr = StackReg(get_sr_mode(mode))

    if ext_file:
        ext_stack = load_stack(ext_file)
        ref = ext_stack[ext_idx]
    else:
        ref = stack[ref_idx]

    aligned = [sr.register_transform(ref, fr) for fr in stack]
    aligned = normalize_stack(np.stack(aligned))
    aligned_frames = [upscale(Image.fromarray(fr)) for fr in aligned]

    path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    tifffile.imwrite(path, aligned, photometric='minisblack')

    return (
        original_frames[0],
        gr.update(value=0, maximum=len(original_frames) - 1),
        aligned_frames[0],
        gr.update(value=0, maximum=len(aligned_frames) - 1),
        path
    )

def reference_align(ref_file, mov_file, mode):
    global ref_frames, mov_frames, reg_frames
    ref_stack = load_stack(ref_file)
    mov_stack = load_stack(mov_file)
    ref_frames = [Image.fromarray(f) for f in ref_stack]
    mov_frames = [Image.fromarray(f) for f in mov_stack]
    sr = StackReg(get_sr_mode(mode))
    aligned = [sr.register_transform(ref_stack[0], f) for f in mov_stack]
    aligned = normalize_stack(np.stack(aligned))
    reg_frames = [upscale(Image.fromarray(f)) for f in aligned]
    path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    tifffile.imwrite(path, aligned, photometric='minisblack')
    return ref_frames[0], gr.update(value=0, maximum=len(ref_frames)-1), \
           reg_frames[0], gr.update(value=0, maximum=len(reg_frames)-1), path

def frame_to_frame_align(file, ref_idx, mov_idx, mode):
    global custom_stack, reg_result
    stack = load_stack(file)
    custom_stack = [Image.fromarray(f) for f in stack]
    sr = StackReg(get_sr_mode(mode))
    aligned = sr.register_transform(stack[ref_idx], stack[mov_idx])
    aligned = normalize_stack(np.stack([aligned]))[0]
    reg_result = Image.fromarray(aligned)
    path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    tifffile.imwrite(path, aligned[np.newaxis, ...], photometric='minisblack')
    return reg_result, path


# Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Pystackreg Web Application")
    gr.Markdown(citation_markdown)

    with gr.Accordion("📘 How to Use This App (Click to Expand)", open=False):
        gr.Markdown(documentation_markdown)

    with gr.Tab("📚 Reference-Based Alignment"):
        with gr.Row():
            file_input = gr.File(label="Upload Stack to Register")
            use_ext_ref = gr.Checkbox(label="Use external reference stack")

        with gr.Row():
            ref_slider = gr.Slider(label="Reference Frame (from uploaded stack)", minimum=0, maximum=0, value=0, step=1, visible=True)
            ext_ref_file = gr.File(label="Upload External Reference Stack (.tif)", visible=False)
            ext_ref_slider = gr.Slider(label="Reference Frame (from external stack)", minimum=0, maximum=0, value=0, step=1, visible=False)

        use_ext_ref.change(
            lambda use_ext: (
                gr.update(visible=not use_ext),
                gr.update(visible=use_ext),
                gr.update(visible=use_ext)
            ),
            use_ext_ref,
            [ref_slider, ext_ref_file, ext_ref_slider]
        )

        ext_ref_file.change(
            lambda f: gr.update(value=0, maximum=len(iio.mimread(f)) - 1) if f else gr.update(value=0, maximum=0),
            ext_ref_file,
            ext_ref_slider
        )

        with gr.Row():
            show_advanced = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                        value="RIGID_BODY", label="Transformation Mode", visible=False)

        show_advanced.change(lambda v: gr.update(visible=v), show_advanced, mode_dropdown)
        run_btn = gr.Button("▶️ Align Stack")

        with gr.Row():
            original_image = gr.Image(label="Original Frame")
            aligned_image = gr.Image(label="Aligned Frame")

        with gr.Row():
            original_slider = gr.Slider(label="Browse Original Stack", minimum=0, maximum=0, value=0, step=1)
            aligned_slider = gr.Slider(label="Browse Aligned Stack", minimum=0, maximum=0, value=0, step=1)

        download = gr.File(label="Download")

        file_input.change(
            lambda f: gr.update(value=0, maximum=len(iio.mimread(f)) - 1) if f else gr.update(value=0, maximum=0),
            file_input,
            ref_slider
        )

        run_btn.click(
            intra_stack_align,
            [file_input, ref_slider, ext_ref_file, ext_ref_slider, mode_dropdown],
            [original_image, original_slider, aligned_image, aligned_slider, download]
        )

        original_slider.change(lambda i: original_frames[i] if 0 <= i < len(original_frames) else None, original_slider, original_image)
        aligned_slider.change(lambda i: aligned_frames[i] if 0 <= i < len(aligned_frames) else None, aligned_slider, aligned_image)

        reset_btn = gr.Button("🔄 Reset Tab")
        reset_btn.click(
            reset_intra_stack,
            outputs=[
                file_input, ref_slider, ext_ref_file, ext_ref_slider,
                original_image, original_slider, aligned_image, aligned_slider, download
            ]
        )

    with gr.Tab("🎯 Stack-Based Alignment"):
        with gr.Row():
            ref_input = gr.File(label="Reference Stack")
            mov_input = gr.File(label="Moving Stack")

        with gr.Row():
            show_adv_ref = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown_ref = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                            value="RIGID_BODY", label="Transformation Mode", visible=False)

        show_adv_ref.change(lambda v: gr.update(visible=v), show_adv_ref, mode_dropdown_ref)
        ref_btn = gr.Button("▶️ Register")

        with gr.Row():
            ref_image = gr.Image(label="Reference Frame")
            reg_image = gr.Image(label="Registered Frame")

        with gr.Row():
            ref_slider = gr.Slider(label="Browse Ref", minimum=0, maximum=0, value=0, step=1)
            reg_slider = gr.Slider(label="Browse Reg", minimum=0, maximum=0, value=0, step=1)

        download_ref = gr.File(label="Download")

        ref_btn.click(reference_align, [ref_input, mov_input, mode_dropdown_ref],
                      [ref_image, ref_slider, reg_image, reg_slider, download_ref])
        ref_slider.change(lambda i: ref_frames[i] if 0 <= i < len(ref_frames) else None, ref_slider, ref_image)
        reg_slider.change(lambda i: reg_frames[i] if 0 <= i < len(reg_frames) else None, reg_slider, reg_image)

        reset_ref_btn = gr.Button("🔄 Reset Tab")
        reset_ref_btn.click(
            reset_reference_based,
            outputs=[ref_input, mov_input, ref_image, ref_slider, reg_image, reg_slider, download_ref]
        )


    with gr.Tab("🧩 Frame-to-Frame Alignment"):
        with gr.Row():
            frame_file = gr.File(label="Upload Stack")
            ref_idx = gr.Slider(label="Reference Frame", minimum=0, maximum=0, value=0, step=1)
            mov_idx = gr.Slider(label="Moving Frame", minimum=0, maximum=0, value=0, step=1)

        with gr.Row():
            show_adv_ftf = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown_ftf = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                            value="RIGID_BODY", label="Transformation Mode", visible=False)

        show_adv_ftf.change(lambda v: gr.update(visible=v), show_adv_ftf, mode_dropdown_ftf)
        frame_btn = gr.Button("▶️ Register Frame")
        frame_output = gr.Image(label="Registered Output")
        download_ftf = gr.File(label="Download")

        frame_file.change(
            lambda f: [gr.update(value=0, maximum=len(iio.mimread(f)) - 1)]*2 if f else [gr.update(value=0, maximum=0)]*2,
            frame_file, [ref_idx, mov_idx]
        )
        frame_btn.click(frame_to_frame_align, [frame_file, ref_idx, mov_idx, mode_dropdown_ftf], [frame_output, download_ftf])

        reset_ftf_btn = gr.Button("🔄 Reset Tab")
        reset_ftf_btn.click(
            reset_frame_to_frame,
            outputs=[frame_file, ref_idx, mov_idx, frame_output, download_ftf]
        )

if __name__ == "__main__":
    demo.launch()
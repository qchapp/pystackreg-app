import gradio as gr
import numpy as np
from PIL import Image
from pystackreg import StackReg
import imageio.v2 as iio
import tifffile
import tempfile
import urllib.request

from core.utils import *

# Globals
original_frames, aligned_frames = [], []
ref_frames, mov_frames, reg_frames = [], [], []
custom_stack, reg_result = [], None

# Reset functions
def reset_intra_stack():
    global original_frames, aligned_frames
    original_frames, aligned_frames = [], []
    return [None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0),
            None, gr.update(value=0, minimum=0, maximum=0), None, gr.update(value=0, minimum=0, maximum=0), None]

def reset_reference_based():
    global ref_frames, mov_frames, reg_frames
    ref_frames, mov_frames, reg_frames = [], [], []
    return [None, None, None, gr.update(value=0, minimum=0, maximum=0),
            None, gr.update(value=0, minimum=0, maximum=0), None]

def reset_frame_to_frame():
    global custom_stack, reg_result
    custom_stack, reg_result = [], None
    return [None, gr.update(value=0, minimum=0, maximum=0),
            gr.update(value=0, minimum=0, maximum=0), None, None]

# Registration logic
def intra_stack_align(f, ref_idx, ext_file, ext_idx, mode):
    global original_frames, aligned_frames
    stack = load_stack(f)
    original_frames = [Image.fromarray(fr) for fr in stack]
    sr = StackReg(get_sr_mode(mode))

    ref = load_stack(ext_file)[ext_idx] if ext_file else stack[ref_idx]
    aligned = [sr.register_transform(ref, fr) for fr in stack]
    aligned = normalize_stack(np.stack(aligned))
    aligned_frames = [upscale(Image.fromarray(fr)) for fr in aligned]

    path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    tifffile.imwrite(path, aligned, photometric='minisblack')

    return (
        original_frames[0], gr.update(value=0, maximum=len(original_frames)-1),
        aligned_frames[0], gr.update(value=0, maximum=len(aligned_frames)-1), path
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

# URL loading
def load_from_url(request: gr.Request):
    params = request.query_params
    if "file_url" in params:
        try:
            url = params["file_url"]
            tmp_path = tempfile.mktemp(suffix=".tif")
            urllib.request.urlretrieve(url, tmp_path)
            return [gr.update(value=tmp_path)]
        except Exception as e:
            print(f"[URL load failed] {e}")
    return [None]

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
            [ref_slider, ext_ref_file, ext_ref_slider]
        )

        ext_ref_file.change(
            lambda f: gr.update(value=0, maximum=len(iio.mimread(f)) - 1) if f else gr.update(value=0, maximum=0),
            ext_ref_file,
            ext_ref_slider
        )

        with gr.Row():
            show_adv = gr.Checkbox(label="Show Advanced Settings", value=False)
            mode_dropdown = gr.Dropdown(["TRANSLATION", "RIGID_BODY", "SCALED_ROTATION", "AFFINE", "BILINEAR"],
                                        value="RIGID_BODY", visible=False, label="Transformation Mode")

        show_adv.change(lambda v: gr.update(visible=v), show_adv, mode_dropdown)
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

        original_slider.change(lambda i: original_frames[i] if 0 <= i < len(original_frames) else None,
                               original_slider, original_image)
        aligned_slider.change(lambda i: aligned_frames[i] if 0 <= i < len(aligned_frames) else None,
                              aligned_slider, aligned_image)

        gr.Button("🔄 Reset Tab").click(
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

        gr.Button("🔄 Reset Tab").click(
            reset_reference_based,
            outputs=[ref_input, mov_input, ref_image, ref_slider, reg_image, reg_slider, download_ref]
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

        show_adv_ftf.change(lambda v: gr.update(visible=v), show_adv_ftf, mode_dropdown_ftf)
        frame_btn = gr.Button("▶️ Register Frame")
        frame_output = gr.Image(label="Registered Output")
        download_ftf = gr.File(label="Download")

        frame_file.change(
            lambda f: [gr.update(value=0, maximum=len(iio.mimread(f)) - 1)] * 2 if f else [gr.update(value=0, maximum=0)] * 2,
            frame_file, [ref_idx, mov_idx]
        )

        frame_btn.click(frame_to_frame_align,
                        [frame_file, ref_idx, mov_idx, mode_dropdown_ftf],
                        [frame_output, download_ftf])

        gr.Button("🔄 Reset Tab").click(
            reset_frame_to_frame,
            outputs=[frame_file, ref_idx, mov_idx, frame_output, download_ftf]
        )

    @demo.load(
        outputs=[file_input, ref_slider, frame_file, ref_idx, mov_idx, ref_input, mov_input]
    )
    def load_from_query(request: gr.Request):
        params = request.query_params
        results = [None] * 7  # 7 outputs

        # One-stack file case (for ref-based + frame-to-frame)
        if "file_url" in params:
            try:
                url = params["file_url"]
                tmp_path = tempfile.mktemp(suffix=".tif")
                urllib.request.urlretrieve(url, tmp_path)
                stack = iio.mimread(tmp_path)
                max_frame = len(stack) - 1

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
                tmp_path_1 = tempfile.mktemp(suffix=".tif")
                tmp_path_2 = tempfile.mktemp(suffix=".tif")
                urllib.request.urlretrieve(params["file_url_1"], tmp_path_1)
                urllib.request.urlretrieve(params["file_url_2"], tmp_path_2)
                results[5] = tmp_path_1  # ref_input
                results[6] = tmp_path_2  # mov_input
            except Exception as e:
                print(f"[Error loading file_url_1 or file_url_2] {e}")

        return results



if __name__ == "__main__":
    demo.launch()
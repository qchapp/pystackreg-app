from pystackreg import StackReg
import numpy as np
import imageio.v2 as iio
from PIL import Image

def get_sr_mode(mode_str): return {
    "TRANSLATION": StackReg.TRANSLATION,
    "RIGID_BODY": StackReg.RIGID_BODY,
    "SCALED_ROTATION": StackReg.SCALED_ROTATION,
    "AFFINE": StackReg.AFFINE,
    "BILINEAR": StackReg.BILINEAR,
}.get(mode_str, StackReg.RIGID_BODY)

def normalize_stack(stack):
    norm_stack = []
    for frame in stack:
        f = frame.astype(np.float32)
        low, high = np.percentile(f, (1, 99))
        f = np.clip(f, low, high)
        f = (f - f.min()) / (np.ptp(f) + 1e-8) * 255 if np.ptp(f) > 0 else np.zeros_like(f)
        norm_stack.append(f.astype(np.uint8))
    return np.stack(norm_stack)

def upscale(image, factor=3):
    return image.resize((image.width * factor, image.height * factor), Image.NEAREST)

def load_stack(file):
    stack = np.array(iio.mimread(file))
    if stack.ndim == 4 and stack.shape[-1] == 3:
        stack = np.mean(stack, axis=-1)
    return normalize_stack(stack)
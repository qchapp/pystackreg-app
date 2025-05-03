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


##### Markdowns #####

citation_markdown = """
        ### 📘 Credits & Acknowledgments

        Register TIFF stacks using multiple alignment strategies.

        **App Author**: [Quentin Chappuis](https://github.com/qchapp)  
        **Pystackreg Author**: [Gregor Lichtenberg](https://github.com/glichtner)  
        🔗 [Pystackreg on GitHub](https://github.com/glichtner/pystackreg)  

        **Original Algorithm Author**: Philippe Thévenaz (EPFL)  
        The core algorithm was originally developed by Philippe Thévenaz and is described in the following publication:

        > P. Thévenaz, U.E. Ruttimann, M. Unser  
        > *A Pyramid Approach to Subpixel Registration Based on Intensity*  
        > *IEEE Transactions on Image Processing*, vol. 7, no. 1, pp. 27–41, January 1998.  
        > 🔗 [View paper](http://bigwww.epfl.ch/publications/thevenaz9801.html)
        """


documentation_markdown = """
        ### Overview

        This app provides three registration modes for 2D TIFF image stacks using the `pystackreg` library.

        ---

        ### 📚 Tab 1: Reference-Based Alignment

        Register a stack using a reference frame. You can:
        - Use a frame from the **same stack** as reference
        - Or upload an **external reference stack**

        1. Upload the stack you want to align.
        2. (Optional) Check "Use external reference stack" to align to a frame from another file.
        3. Choose the reference frame using the slider.
        4. (Optional) Choose transformation mode.
        5. Click **▶️ Align Stack**.
        6. Use sliders to browse original/aligned results and download the output.

        ---

        ### 🎯 Tab 2: Stack-Based Alignment

        Align one stack (moving) to another (reference).

        1. Upload both **reference** and **moving** stacks.
        2. (Optional) Choose transformation mode.
        3. Click **▶️ Register** to align.
        4. Browse and download registered stack.

        ---

        ### 🧩 Tab 3: Frame-to-Frame Alignment

        Align a **single frame to another** from the same stack.

        1. Upload a stack.
        2. Select **reference** and **moving** frames using sliders.
        3. Choose transformation mode.
        4. Click **▶️ Register Frame**.
        5. View/download the result.

        ---

        ### 🔄 Reset Buttons

        Each tab includes a **Reset Tab** button that clears inputs, outputs, and internal state.

        ---

        ### 🧠 Credits

        App developed by **Quentin Chapuis**  
        Library: [`pystackreg`](https://github.com/glichtner/pystackreg) by **Georg Lichtenberg**
        """
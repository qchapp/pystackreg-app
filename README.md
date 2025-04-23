# 🧠 Stack Image Registration Web App  
A web-based application for image stack registration powered by **Gradio** and **pystackreg**.  
This tool allows users to align and stabilize multi-frame TIFF images using a variety of transformation models.

<p align="center">
    <img src="images/app.png" height="500">
</p>

---

## 🚀 Try the App  
The application is running on [Hugging Face](https://huggingface.co/), try it using this [link](https://huggingface.co/spaces/your-username/pystackreg-app)!

---

## 🛠️ Installation  
We recommend performing the installation in a clean Python environment.

This app requires `python>=3.10`. To install dependencies, run:

```sh
pip install -r requirements.txt
```

---

## ▶️ Usage  
To run the app locally:

```sh
python app.py
```

Then open your browser and go to: [http://localhost:7860](http://localhost:7860)

---

## 🔍 About Stack Registration  
This app uses the [pystackreg](https://github.com/glichtner/pystackreg) library, a Python port of the TurboReg/StackReg algorithms.  
It supports several transformation models for alignment:
- Translation
- Rigid Body
- Scaled Rotation
- Affine
- Bilinear

---

## 📂 Features  
This application provides three core registration modes:

1. **📚 Reference-Based Alignment**  
   Align all frames within a stack to a selected reference frame — either from the same stack or an external 3D image.

2. **🎯 Stack-Based Alignment**  
   Align every frame in one stack to the first frame of another reference stack.

3. **🧩 Frame-to-Frame Alignment**  
   Align a single frame to another frame within the same stack.

By default, the app uses the **Rigid Body** transformation mode for all alignment tasks.  
If needed, users can enable **Advanced Settings** in each tab to select from other transformation models, such as Translation, Affine, or Bilinear.

Each mode offers:
- 🔍 Interactive image preview
- 🧭 Frame-by-frame navigation
- 💾 Downloadable aligned results
- ⚙️ Customizable transformation models via advanced options

---

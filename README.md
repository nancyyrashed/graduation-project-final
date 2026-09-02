# GraduationProject: Neural Style Transfer Platform for Real-Time Image and Video Stylization

This project was developed as my final-year project for the BSc Computer Science programme at Goldsmiths, University of London, where it received a **Distinction** grade.

It investigates the trade-offs between image quality, style fidelity, flexibility, and computational efficiency across several state-of-the-art Neural Style Transfer (NST) approaches — going beyond textbook implementations by introducing enhanced architectures, multi-style transfer, and video stylization with temporal consistency.

**Live Demo:** https://neural-style-transfer-graduation-25.streamlit.app/

**Full Report:** [`final report grad project.pdf`](final%20report%20grad%20project.pdf)

---

## Objectives

* Investigate the strengths and limitations of existing Neural Style Transfer methods.
* Improve the scalability and efficiency of NST for real-time applications.
* Compare multiple NST architectures using qualitative and quantitative evaluation metrics.
* Extend style transfer capabilities to support:
  * Arbitrary style transfer
  * Multi-style transfer
  * Real-time video style transfer

---

## Implemented Models

| Model | Description | Notebook |
|---|---|---|
| **Gatys et al. (2016)** | The original optimization-based NST approach — iteratively optimizes a generated image against VGG-19 content/style feature losses. Highest style fidelity, but slow (no feed-forward inference). | `gatys_nst_final.ipynb` |
| **Johnson et al. (2016)** | A feed-forward network trained per-style for real-time image stylization. | `johnson_nst_final.ipynb` |
| **AdaIN** | Adaptive Instance Normalization — an arbitrary style transfer model that applies unseen artistic styles at inference time without retraining. | `adain_nst_final.ipynb` |
| **Dumoulin et al. (2017)** | Conditional Instance Normalization (CIN) — a single feed-forward network supporting multiple learned styles via style codes. | `dumoulin_nst_final.ipynb` |
| **Dumoulin V2** *(proposed extension)* | Combines CIN and AdaIN so the model generalizes to unseen styles at runtime, rather than being limited to a fixed set of learned style codes. | `dumoulin_v2_nst_final.ipynb` |
| **Dumoulin V2 Multi-Style** *(extension)* | Applies multiple artistic styles simultaneously within a single image, building on Dumoulin V2. | `dumoulin_v2_multi-style-nst.ipynb` |
| **Dumoulin V2 Video Transfer** *(extension)* | Applies the Dumoulin V2 model frame-by-frame to video, aiming for temporal consistency and reduced flickering. | `dumoulin_v2_video-st_final.ipynb` |

---

## Repository Structure

```
graduation-project-final/
├── gatys_nst_final.ipynb              # Gatys et al. implementation + experiments
├── johnson_nst_final.ipynb            # Johnson et al. implementation + experiments
├── adain_nst_final.ipynb              # AdaIN implementation + experiments
├── dumoulin_nst_final.ipynb           # Dumoulin (CIN) implementation + experiments
├── dumoulin_v2_nst_final.ipynb        # Dumoulin V2 (CIN + AdaIN) implementation
├── dumoulin_v2_multi-style-nst.ipynb  # Multi-style extension of Dumoulin V2
├── dumoulin_v2_video-st_final.ipynb   # Video stylization extension of Dumoulin V2
├── final report grad project.pdf      # Full written dissertation/report
│
├── gatys/           # Per-experiment stylized outputs + intermediate iterations
├── johnson/         # Per-experiment trained models (.pth), loss CSVs, outputs
├── adain/           # Per-experiment trained models (.pth), loss CSVs, outputs
├── dumoulin/        # Per-experiment trained models (.ckpt), loss CSVs, outputs
├── dumoulin_v2/               # Trained model + outputs for the CIN+AdaIN extension
├── dumoulin_v2_multi-style/   # Trained model + multi-style outputs
├── dumoulin_v2_video-st/      # Trained model + stylized video outputs
│
├── portfolio/       # Curated final result images/videos per model, for the report/demo
│
└── full_nst_website/           # Streamlit web application (the live demo)
    ├── full_app.py              # Main Streamlit app — model selection + inference UI
    ├── requirements.txt         # Python dependencies for the web app
    ├── packages.txt             # System packages (ffmpeg, libgl1-mesa-glx) for deployment
    ├── *.pth / *.ckpt           # Pretrained model weights bundled with the app
    └── .devcontainer/           # Dev Container config for one-click Codespaces setup
```

Each experiment folder (`*_results_expN/`) generally contains:
- The trained model checkpoint (`.pth` or `.ckpt`)
- A `losses.csv` / `weighted_losses.csv` tracking loss curves over training
- Stylized output image(s), including intermediate iterations where relevant

---

## Web Application

A **Streamlit** web app (`full_nst_website/full_app.py`) lets users upload content and style images, choose an NST method, and generate stylized outputs interactively. The deployed app exposes:

- ADAIN
- Dumoulin
- Dumoulin V2
- Dumoulin V2 Multi-Style
- Dumoulin V2 Video
- Johnson

(Gatys is not included in the live app, since it's iterative/optimization-based and too slow for interactive use — it's demonstrated in its own notebook instead.)

### Running the web app locally

```bash
cd full_nst_website
pip install -r requirements.txt
streamlit run full_app.py
```

On Linux/deployment environments, also install the system packages listed in `packages.txt` (`ffmpeg`, `libgl1-mesa-glx`) — required for OpenCV and video/GIF conversion.

A `.devcontainer` config is included, so the app can also be opened directly in a GitHub Codespace, which will install dependencies and launch Streamlit automatically.

### Running the notebooks

Each `*_nst_final.ipynb` notebook is self-contained (imports, model definition, training/inference, and evaluation) and was developed in Google Colab. Core dependencies across the notebooks:

```bash
pip install torch torchvision pillow numpy scipy opencv-python matplotlib seaborn imageio imageio-ffmpeg
```

---

## Technologies Used

**Languages:** Python

**Deep Learning:** PyTorch, TensorFlow, Keras

**Computer Vision / Image Processing:** OpenCV, Pillow (PIL), ImageIO

**Data Analysis & Visualization:** NumPy, Matplotlib, Seaborn

**Deployment & Development:** Streamlit, GitHub, Google Colab, FFmpeg

---

## Evaluation Metrics

Models were evaluated using:

* Content Loss
* Style Loss
* SSIM (Structural Similarity Index)
* LPIPS (Learned Perceptual Image Patch Similarity)
* Processing Time
* User surveys and qualitative evaluation

---

## Results

The project demonstrated clear trade-offs between image quality, flexibility, and computational efficiency across the different NST architectures:

* **Gatys et al.** produced the highest style fidelity but required significant per-image optimization time.
* **Johnson et al.** enabled real-time stylization through a feed-forward network, at the cost of being fixed to one style per trained model.
* **AdaIN** provided fast arbitrary style transfer without retraining, generalizing to unseen styles.
* **Dumoulin-based architectures** achieved improved flexibility across multiple learned styles.
* The proposed **Dumoulin V2 extensions** successfully supported unseen styles, multi-style transfer, and video stylization while maintaining strong visual quality.

Sample outputs for each model/experiment are in the corresponding results folders (`gatys/`, `johnson/`, `adain/`, `dumoulin/`, `dumoulin_v2*/`) and curated comparisons are in `portfolio/`.

---

## Notes

- Trained model weights (`.pth`, `.ckpt`) are committed directly in the repo for reproducibility of the reported results; if cloning for development, be aware this makes the repo large.
- The full methodology, related work, quantitative results, and discussion are documented in `final report grad project.pdf`.

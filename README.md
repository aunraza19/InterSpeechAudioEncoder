# Quad-Core Hybrid Encoder (QCHE)   
**Interspeech 2026 Audio Encoder Capability Challenge Submission**

This repository contains the implementation of the **Quad-Core Hybrid Encoder (QCHE)**, a lightweight representation model designed for Interspeech 2026.  
Our model utilizes a **Convolutional-Transformer hybrid architecture** to achieve high-fidelity audio embeddings (128-dimensional) optimized for downstream **Large Audio Language Models (LALMs)**.

---

## 1. Model Weights

The trained checkpoint is not stored in this repository because it exceeds GitHub's file-size limit. You can download it from Google Drive here:

[Download `best_model.pth`](https://drive.google.com/file/d/1tvrCRhHF30KlZ3eHZk6LwjXXm2MPzODI/view?usp=sharing)

### Install the model

1. Download the file from the link above.
2. Save it in the root of this repository.
3. Rename the file to `best_model.pth`.

The encoder loads weights from `best_model.pth` by default, so no code changes are needed if the file is placed in the project root. If you want to use a different location, pass a custom path when creating `MyEncoder`.

---

## 2. Key Features

- **Architecture:**  
  4-stage 1D-CNN temporal compressor followed by a 6-layer Transformer Encoder.

- **Temporal Resolution:**  
  25 Hz (40 ms hop size), aligning with competition standards.

- **Efficiency:**  
  Approximately 3.5M parameters, designed to run under the 24 GB VRAM constraint.

- **Performance:**  
  Achieved approximately **97.5% Log-Mel Cosine Similarity** on speech reconstruction tasks.


---

## 3. Dataset Composition: *"The Quad-Core Mix"*

Our model was trained on a balanced curriculum of four publicly accessible datasets to ensure robust generalization across:

- **Track A:** Classification  
- **Track B:** Understanding

| Source               | Weight | Role            | Dataset Description                          |
|----------------------|--------|------------------|----------------------------------------------|
| Emilia-YODAS         | 40%    | Foundation       | Diverse speech data for accent invariance.   |
| LibriSpeech          | 30%    | Precision        | High-fidelity phoneme representations.       |
| ESC-50               | 15%    | Event Detection  | Environmental sound classification features.|
| Zenodo (King-ASR)    | 15%    | Ambience         | Robustness against background noise.         |

---
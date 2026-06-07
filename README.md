
# Jigsaw Puzzle Reconstruction

An end-to-end deep neural pipeline implemented in TensorFlow/Keras to solve the problem of visual puzzle reconstruction. The project takes an input image that has been segmented, scrambled, and eroded, resolves its structural sequence, inpaints the missing boundary gaps, and restores the original picture.

## 📋 Problem Statement
Given 9 scattered visual patches of size `(28×28×3)` derived from a `3×3` grid segment of an original `96×96` RGB image—where each individual patch has had its outer 2-pixel borders completely eroded:
* **Infer** the authentic spatial organization matrix of the scrambled patches.
* **Inpaint** the structural 2-pixel border gap intervals introduced between matching edges.
* **Reconstruct** a globally coherent unified image matching the target dimensions.

---

🧠 System Architecture & Design
The configuration leverages a state-of-the-art neural architecture combining standard convolutional layers, self-attention sequences, permutation estimation operators, and iterative segmentation heads, yielding an end-to-end differentiable setup.


```
            +-------------------+
            |  9 Eroded Patches |
            +---------+---------+
                      |
                      v
           +----------+----------+
           |  Shared CNN Encoder |
           +----------+----------+
                      |
                      v
         +------------+------------+
         |    Transformer Encoder  |
         +------------+------------+
                      |
                      v
        +-------------+-------------+
        |    Sinkhorn Ordering Head |
        +-------------+-------------+
                      |
                      v
       +--------------+--------------+
       | Differentiable Patch Assm.  |
       +--------------+--------------+
                      |
                      v
          +-----------+-----------+
          |    U-Net Refinement   |
          +-----------+-----------+
                      |
                      v
          +-----------+-----------+
          |   Reconstructed Image |
          +-----------------------+

```


1. **Shared CNN Encoder:** Maps each input patch independently into a localized `256-dimensional` embedded vector. Weight-sharing handles strict input permutation equivariance.
2. **Transformer Encoder:** Utilizes a `4-layer` multi-head self-attention module over the 9 embedded patch tokens to model absolute global context across discrete grid dependencies.
3. **Sinkhorn Ordering Head:** Maps the sequence embeddings to approximate a doubly-stochastic assignment mapping. This generates a differentiable soft-permutation matrix optimized via standard loss updates.
4. **Differentiable Patch Assembler:** Performs a matrix operation to loosely place input fragments onto the output `96×96` canvas space according to the generated layout map, initialization gaps are padded with zeros.
5. **U-Net Refinement Network:** A deep U-Net structure featuring multi-scale feature synthesis maps over the raw unaligned composite canvas. The model learns the boundary residual map (the missing detail updates) instead of reconstructing the raw scene map from scratch, ensuring fast, stable loss optimization.



## 📊 Summary of Specifications

| Property / Parameter | Details | Status |
| :--- | :--- | :--- |
| **Deep Learning Framework** | TensorFlow 2.x / Keras | Keras-Native |
| **Total Parameter Count** | **5,509,260** | Within `< 6,000,000` limit ✓ |
| **Pretrained Network Weights** | None | Trained entirely from scratch ✓ |
| **Objective / Loss Metric** | Mean Absolute Error (MAE) | Applied globally on `96×96` |
| **Dataset Source** | STL-10 Unlabeled Dataset | 100,000 source images |

---

## 📂 Project Structure Framework

```text
├── dataset/
│   └── STL-10 Binary (Extracted locally via utility script)
├── DeepLearning_Project_Mahdi_Zavvar.ipynb   # Complete execution workbook
├── README.md                                 # Project documentation
└── weights/                                  # Trained pipeline weights

```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment

Ensure you have Python 3.10+ and a compute environment enabled with a modern GPU accelerator (e.g., NVIDIA T4).

```bash
pip install numpy tensorflow keras matplotlib

```

### 2. Dataset Initialization

The application utilizes the **STL-10 Unlabeled Dataset** (containing 100,000 target arrays of resolution `96×96×3`). The dataset logic uses a memory-mapped array setup (`np.memmap`) to loop image inputs gracefully under constraints without taxing memory overhead boundaries.

```python
# Automatic pipeline retrieval setup built into the model
from tensorflow.keras.utils import PyDataset
import numpy as np

# Grid transformations create random indices for the patch pipeline
# Each training vector processes 9 discrete cropped images

```

### 3. Pipeline Model Pipeline Check

You can evaluate your structural pipeline mapping blocks by performing summary network verification checks directly via:

```python
# Instantiate model framework via code definitions inside workbook
model.summary()

```

---

## 📈 Key Technical Innovations

* **Permutation Equivariance:** Utilizing identical visual feature spaces inside the encoder means spatial patch locations are assigned purely by evaluating context features, freeing the pipeline from ordering bias.
* **End-to-End Matching Execution:** The implementation avoids rigid non-differentiable matching operations (such as Hungarian Matching algorithms). Instead, an elegant, continuous matrix operation relies on structural **Sinkhorn Iterations** to smoothly propagate errors end-to-end.
* **Residual Canvas Inpainting:** The U-Net structure calculates internal pixel adjustments dynamically over unaligned borders rather than mapping raw scene components, speeding optimization steps significantly.

---

### 🎓 Academic Affiliation & Coursework Info
* **Institution:** Alma Mater Studiorum – Università di Bologna (University of Bologna)
* **Degree Program:** Master's in Electronics Engineering for Intelligent Systems, Big Data and IoT
* **Course:** Deep Learning
* **Teacher**: Andrea Asperti
* **Tutor**: Fabio Merizzi
* **Tutor**: Beniamino Tartufoli
* **Author:** Mahdi Zavvar ([Github](https://github.com/M-zavvar/deepLearning.git))

### ⚖️ License
This project code and associated materials are licensed under the **MIT License**.

```text
Copyright (c) 2026 Mahdi Zavvar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
...

```

```

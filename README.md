# LungNoduleNet — 3D CT Pulmonary Nodule Detection

### An AI-Assisted Framework for Pulmonary Nodule Detection from Volumetric CT Scans

LungNoduleNet is a deep learning pipeline for automated pulmonary nodule detection from 3D chest CT scans. The project uses a 3D Convolutional Neural Network (3D-CNN) to learn spatial features across volumetric CT data and classify candidate regions as nodule or non-nodule.

The model is trained and evaluated on the LUNA16 (LUng Nodule Analysis 2016) dataset using 3D CT patches of size `64 × 64 × 64`, with 10-fold cross-validation for evaluation.

## Highlights

- 3D CNN for volumetric pulmonary nodule classification
- Trained on 4,032 LUNA16 samples
- 64 × 64 × 64 3D CT patches
- 10-fold cross-validation
- Weighted Cross-Entropy for class imbalance
- PyTorch-based training pipeline
- 97.67% average 10-fold accuracy
- 0.9954 ROC-AUC
- Precision, recall, F1-score, accuracy, and confusion-matrix analysis

## Problem Statement

Pulmonary nodules can be difficult to identify in volumetric CT scans, particularly when they are small or surrounded by complex anatomical structures.

Manual examination of complete CT volumes is time consuming and can be affected by observer fatigue. LungNoduleNet explores the use of 3D deep learning as an AI assisted screening tool to identify candidate pulmonary nodules.

### Objective

The primary objective is to develop a robust 3D CNN capable of distinguishing:

- Nodule
- Non-Nodule

from candidate regions extracted from volumetric chest CT scans.

> Note: This project performs nodule/non-nodule classification. It does not perform benign/malignant cancer diagnosis or malignancy grading.

## Dataset

### LUNA16

The project uses the LUNA16 (LUng Nodule Analysis 2016) benchmark dataset.

| Property | Value |
|---|---:|
| CT scans | 888 |
| Total samples | 4,032 |
| Positive nodule samples | 1,086 |
| Negative samples | 2,946 |
| Dataset subsets | 0–9 |
| Input volume | 64 × 64 × 64 |

The complete LUNA16 subsets 0–9 were utilized for volumetric learning.

Radiologist-provided annotations were used to identify candidate nodule coordinates and corresponding labels.

### Class Distribution

- Positive: 1,086 nodule samples
- Negative: 2,946 non-nodule samples

Because of the class imbalance, weighted cross-entropy was used during training.

## Data Representation

Instead of treating CT scans as independent 2D images, the project preserves their volumetric structure.

Each candidate region is represented as a:

```text
64 × 64 × 64
```

3D CT cube.

The reported dataset representations are:

```text
Raw dataset:
(4032, 64, 64, 64)

Labels:
(4032,)

Normalized dataset:
(4032, 1, 64, 64, 64)
```

This allows the model to learn spatial information within CT volumes and across adjacent slices.

## System Pipeline

```text
                 ┌──────────────────────┐
                 │    LUNA16 CT Scans   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ CT Preprocessing     │
                 │ & Normalization      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ 3D Cube Extraction   │
                 │   64 × 64 × 64       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       3D CNN         │
                 │                      │
                 │ Conv3D               │
                 │ BatchNorm            │
                 │ ReLU                 │
                 │ MaxPool3D            │
                 │        × 3            │
                 │                      │
                 │ Flatten              │
                 │ Fully Connected      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       Softmax        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Nodule / Non-Nodule  │
                 └──────────────────────┘
```

## Model Architecture

LungNoduleNet uses a 3D Convolutional Neural Network designed to process volumetric CT patches.

### Architecture

```text
Input: 1 × 64 × 64 × 64

        ↓

Conv3D
BatchNorm
ReLU
MaxPool3D

        ↓

Conv3D
BatchNorm
ReLU
MaxPool3D

        ↓

Conv3D
BatchNorm
ReLU
MaxPool3D

        ↓

Flatten

        ↓

Fully Connected Layer

        ↓

Softmax

        ↓

Nodule / Non-Nodule
```

### Why 3D CNN?

A conventional 2D CNN processes individual slices independently.

A 3D CNN applies convolution across:

```text
Height × Width × Depth
```

allowing the model to learn volumetric spatial patterns across multiple CT slices.

## Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss | Weighted Cross-Entropy |
| Epochs | 18 |
| Validation | 10-Fold Cross-Validation |
| Hardware | 2 × NVIDIA Tesla T4 |
| LR Scheduling | Enabled |

A learning-rate scheduling strategy was used for stable convergence during training.

## Evaluation Strategy

### 10-Fold Cross-Validation

The dataset was evaluated using 10-fold cross-validation.

Each fold serves as the validation/test portion once while the remaining folds are used for training.

```text
Fold 1  → Train / Validate
Fold 2  → Train / Validate
Fold 3  → Train / Validate
...
Fold 10 → Train / Validate
```

The reported average accuracy across the 10 folds was:

```text
97.67%
```

## Results

### Overall Performance

| Metric | Result |
|---|---:|
| Average 10-Fold Accuracy | 97.67% |
| ROC-AUC | 0.9954 |

The reported ROC-AUC of 0.9954 indicates strong separation between nodule and non-nodule samples on the evaluated data.

### Fold 10 Classification Performance

| Metric | Score |
|---|---:|
| Precision | 97.17% |
| Recall / Sensitivity | 95.37% |
| F1-Score | 96.26% |
| Accuracy | 98.01% |
| ROC-AUC | 0.9954 |

### Fold 10 Confusion Matrix

```text
                 Predicted
              Negative  Positive

Actual Negative    292       3

Actual Positive      5     103
```

Therefore:

```text
True Negatives  = 292
True Positives  = 103
False Positives = 3
False Negatives = 5
```

## Evaluation Visualizations

The project includes:

- Confusion matrix
- Precision, recall, and F1-score
- ROC curve
- ROC-AUC analysis
- Prediction analysis

If visualizations are added to the repository, they can be organized as:

```text
assets/
├── architecture.png
├── confusion_matrix.png
├── classification_report.png
└── roc_curve.png
```

## Project Structure

```text
3D-CNN-Lung-Nodule-Detection/
│
├── lung_nodule_detection_3dcnn.ipynb
├── README.md
├── assets/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   ├── classification_report.png
│   └── roc_curve.png
│
└── .gitignore
```

## Tech Stack

### Deep Learning
- PyTorch
- 3D Convolutional Neural Networks

### Data Processing
- NumPy
- CT volume preprocessing
- 3D patch extraction
- Data normalization

### Evaluation
- Scikit-learn
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Dataset
- LUNA16

### Hardware
- NVIDIA Tesla T4 × 2

## Key Technical Components

### 1. Volumetric Representation

CT data is represented as 3D volumes instead of independent 2D slices, allowing the model to exploit depth-wise spatial information.

### 2. 3D Patch Extraction

Candidate coordinates are converted into normalized:

```text
64 × 64 × 64
```

volumetric patches for model input.

### 3. Class-Imbalance Handling

The dataset contains substantially more negative samples than positive samples. Weighted Cross-Entropy was incorporated into the training objective.

### 4. Cross-Validation

10-fold cross-validation was used to obtain a more robust estimate of model performance.

### 5. ROC Analysis

ROC-AUC was used to evaluate the model's ability to distinguish between nodule and non-nodule samples across classification thresholds.

## Limitations

The current implementation has several limitations:

- Binary classification only: nodule vs. non-nodule
- Does not perform benign/malignant prediction
- Does not incorporate patient-level clinical metadata
- Evaluation is limited to the LUNA16 dataset
- No external clinical validation has been performed

The current implementation should therefore be viewed as an academic/research framework rather than a clinically deployed diagnostic system.

## Future Work

Potential extensions include:

### 3D Transformer Architectures

Explore transformer-based 3D architectures such as 3D Vision Transformers for volumetric representation learning.

### Multimodal Learning

Integrate:

```text
CT Imaging
     +
Clinical Metadata
     +
Patient History
```

to provide richer predictive context.

### PET + CT Fusion

Extend the system toward multimodal medical imaging using PET and CT information.

### Clinical Integration

Investigate integration with hospital imaging systems and real-time computer-aided detection workflows.

## Results at a Glance

```text
                    LungNoduleNet

Dataset              LUNA16
CT Scans             888
Samples              4,032
Input                64 × 64 × 64
Model                3D CNN
Validation           10-Fold CV
Framework            PyTorch

Average Accuracy     97.67%
ROC-AUC              0.9954

Fold 10 Precision    97.17%
Fold 10 Recall       95.37%
Fold 10 F1           96.26%
Fold 10 Accuracy     98.01%
```

## Conclusion

LungNoduleNet demonstrates the application of 3D deep learning to volumetric CT analysis for pulmonary nodule detection.

Using the complete LUNA16 subsets 0–9, 3D CT patches, a PyTorch-based 3D CNN, weighted training, and 10-fold cross-validation, the reported results achieved an average accuracy of 97.67% and ROC-AUC of 0.9954.

The project provides a foundation for further work in 3D medical image analysis, multimodal learning, and AI-assisted pulmonary nodule detection.

## Disclaimer

This project is an academic/research implementation and is not intended for clinical diagnosis or medical decision-making. The reported results are based on the LUNA16 dataset and do not constitute evidence of clinical effectiveness or deployment readiness.

## Authors

**Team Qwerty**

- Angelica Das
- Kasukurthi Ananya
- Yash Mohan

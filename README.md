<div align="center">

<br/>

```
 █████╗ ██╗   ██╗ █████╗ ██╗   ██╗███████╗
██╔══██╗██║   ██║██╔══██╗██║   ██║██╔════╝
███████║██║   ██║███████║██║   ██║███████╗
██╔══██║╚██╗ ██╔╝██╔══██║╚██╗ ██╔╝╚════██║
██║  ██║ ╚████╔╝ ██║  ██║ ╚████╔╝ ███████║
╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝  ╚═══╝  ╚══════╝
```

# Audio-Visual Approximation of Video Semantic Space

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Colab](https://img.shields.io/badge/Run%20on-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**[Open in Colab](#quick-start)** · **[View Results](#results--benchmarks)** · **[Architecture](#model-architectures)** · **[Dataset](#dataset)** · **[Report](#technical-report)**

<br/>

</div>

---

##  Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Project Overview](#project-overview) | Research context, motivation, objective |
| 2 | [Pipeline Architecture](#pipeline-architecture) | End-to-end flowchart |
| 3 | [Dataset](#dataset) | AVMIT, TFRecord format, pre-extracted embeddings |
| 4 | [Model Architectures](#model-architectures) | LSR, CMD, Baselines |
| 5 | [Evaluation Framework](#evaluation-framework) | Metrics: CosSim, MSE, R@K, MedR, Latency |
| 6 | [Results & Benchmarks](#results--benchmarks) | Comparative tables |
| 7 | [Visualisations](#visualisations) | PCA, t-SNE, UMAP, Dashboards |
| 8 | [Repository Structure](#repository-structure) | File tree |
| 9 | [Quick Start](#quick-start) | Setup & run instructions |
| 10 | [Configuration](#configuration) | All hyperparameters explained |
| 11 | [Technical Report](#technical-report) | Deliverable link |
| 12 | [References](#references) | Papers & prior art |

---

## Project Overview

### Context

The project is developed for the **AIMS DTU Research Internship 2026** in the domain of multimodal representation learning. It addresses a fundamental efficiency question in video AI: dense temporal processing of video sequences is computationally expensive, yet the semantic content of many videos is largely captured by a representative frame and its audio.

### Objective

To design, train, and evaluate a multimodal representation learning framework that determines whether the **semantic embedding of a complete video sequence** can be accurately approximated using only:

-  A **single static visual frame** (or sparse keyframes) extracted from the video
-  The corresponding **audio track** of the clip

The approximation must be validated against ground-truth full-video embeddings across three primary dimensions: **latent space proximity**, **downstream retrieval performance**, and **computational efficiency**.

### Why This Matters

```
Full Video Processing                          Our Approach
─────────────────────                          ────────────
16–32 frames × temporal model            vs.  1 frame + audio waveform
~500ms inference / sample                vs.  ~5ms inference / sample
GPU + High VRAM requirement              vs.  CPU-deployable
```

This lightweight AV approximation framework matches 80–90% of a full-video model's retrieval performance at 1/100th of the compute. It has potentially significant practical value for real-time video search, on-device inference, and large-scale indexing.

---

## Pipeline Architecture

### End-to-End System Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AVMIT DATASET (.tar)                                │
│                  gzip-compressed SequenceExample TFRecords                  │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 1 — FEATURE EXTRACTION                          │
│                                                                             │
│   ┌───────────────────────────┐      ┌──────────────────────────────────┐   │
│   │     VISUAL ENCODER        │      │        AUDIO ENCODER             │   │
│   │   VGG16 (fc layer)        │      │   VGGish (AudioSet pre-trained)  │   │
│   │   Output: (T, 512-d)      │      │   Output: (T, 128-d)             │   │
│   │   Mean-Pool → (512,)      │      │   Mean-Pool → (128,)             │   │
│   └───────────┬───────────────┘      └──────────────┬───────────────────┘   │
│               │                                      │                       │
│               └──────────────┬───────────────────────┘                      │
│                              ▼                                               │
│                    AV Concat: (640,) ──────────────► Ground-Truth: (512,)   │
│                              │                       [Visual embedding = GT] │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
              ┌────────────────┴───────────────────┐
              ▼                                     ▼
┌─────────────────────────┐           ┌─────────────────────────────────────┐
│  STAGE 2A — UNSUPERVISED│           │    STAGE 2B — SUPERVISED LEARNING   │
│  DIMENSIONALITY REDUCTION│          │                                     │
│                         │           │  ┌───────────────────────────────┐  │
│  Input: AV concat (640) │           │  │  METHOD 1                     │  │
│                         │           │  │  Latent Space Regression      │  │
│  ┌──────┐ ┌──────┐ ┌──┐ │           │  │  FusionMLP: Audio(128) → (512)│  │
│  │ PCA  │ │t-SNE │ │UMAP│           │  │  Loss: 0.5×MSE + 0.5×CosSim  │  │
│  │ 2D/3D│ │ 2D/3D│ │2D/3D          │  └───────────────────────────────┘  │
│  └──────┘ └──────┘ └──┘ │           │                                     │
│  Matplotlib + Plotly HTM │           │  ┌───────────────────────────────┐  │
└─────────────────────────┘           │  │  METHOD 2                     │  │
                                      │  │  Cross-Modal Distillation     │  │
                                      │  │  StudentMLP: AV(640) → (512)  │  │
                                      │  │  Teacher: VGG16 visual probe  │  │
                                      │  │  Loss: KL(T=4.0) + MSE + Cos  │  │
                                      │  └───────────────────────────────┘  │
                                      └──────────────┬──────────────────────┘
                                                     │
                                                     ▼
                              ┌─────────────────────────────────────────────┐
                              │             STAGE 3 — EVALUATION            │
                              │                                             │
                              │  Latent Proximity  │  Retrieval  │ Efficiency│
                              │  ──────────────    │  ─────────  │ ─────────│
                              │  CosSim ↑          │  R@1 ↑      │ Latency ↓│
                              │  MSE ↓             │  R@5 ↑      │ Params ↓ │
                              │                    │  R@10 ↑     │ FLOPs ↓  │
                              │                    │  MedR ↓     │          │
                              └─────────────────────────────────────────────┘
```

### Data Flow Through the Pipeline

| Stage | Input | Operation | Output |
|-------|-------|-----------|--------|
| **TFRecord Parse** | `.tfrecord` (gzip SequenceExample) | TF `SequenceExample` decoder | `audio (T, 128)`, `visual (T, 512)`, `label` |
| **Mean-Pool** | Temporal embeddings per clip | Temporal averaging | `audio (128,)`, `visual (512,)` per sample |
| **Test-Train-Val Split** | Full dataset | 75% / 15% / 10% stratified split | `train`, `val`, `test` splits |
| **Dim Reduction** | AV concat `(640,)` | PCA → t-SNE, UMAP | 2D/3D embedding plots |
| **Regression** | Audio `(128,)` only | FusionMLP cross-modal prediction | Predicted visual `(512,)` |
| **Distillation** | AV concat `(640,)` | Student-teacher soft targets | Student embedding `(512,)` |
| **Evaluation** | Predictions vs GT visual | CosSim, MSE, R@K, MedR, Latency | Comparative results table |
| **Dashboard** | All method results | Matplotlib 3-row comparative plot | `comparative_dashboard.png` |

---

## Dataset

### AVMIT (Audio-Visual MIT Dataset)

This pipeline is built around the **AVMIT(Audio-Visual Moments in Time)** dataset, a large-scale, multi-class, audio-visual benchmark featuring pre-extracted temporal embeddings from action video clips.

| Property | Value |
|----------|-------|
| Visual Encoder | VGG16 (ImageNet pre-trained), `fc` layer output |
| Audio Encoder | VGGish (AudioSet pre-trained) |
| Visual Dim | 512-d per frame |
| Audio Dim | 128-d per frame |
| Format | gzip-compressed `SequenceExample` TFRecords in a `.tar` archive |
| Temporal Depth | Variable-length sequences per clip |
| Pooling Strategy | Mean-pool across time → flat per-clip embedding |

### Accessing the Dataset

The AVMIT pre-extracted features are hosted on Google Drive:

> **[ Download: AVMIT\_VGGish\_VGG16.tar](https://drive.google.com/file/d/1ZwqD_Y2aYOSlkLQLEIJ9PWCWcBZOtC1j/view)**

Or download directly in the notebook:

```python
import gdown
gdown.download(
    'https://drive.google.com/uc?id=1ZwqD_Y2aYOSlkLQLEIJ9PWCWcBZOtC1j',
    'AVMIT_VGGish_VGG16.tar', quiet=False
)
```

### TFRecord Schema

Each `.tfrecord` is a gzip-compressed `SequenceExample` with the following fields:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `filename` | context bytes | scalar | Clip identifier |
| `label_index` | context int64 | scalar | Integer class label |
| `label_name` | context bytes | scalar | Human-readable class name |
| `audio_embedding` | feature list float | `(T, 128)` | VGGish embeddings per frame |
| `rgb` / `visual_embedding` | feature list float | `(T, 512)` | VGG16 fc embeddings per frame |

### Embedding Cache

On first run, parsed embeddings are saved as `.npz` to `./avmit_cache/`. All subsequent runs load from cache instantly — no re-parsing needed.

```
avmit_cache/
  avmit_embeddings.npz      ← audio_embs, visual_embs, label_ids, label_names
```

---

## Model Architectures

### Method 1 — Latent Space Regression (`FusionMLP`)

A cross-modal MLP that predicts the **visual (VGG16) embedding from audio alone**, forcing genuine cross-modal learning with no information leakage.

```
Input:  Audio embedding  (B, 128)   ← VGGish only
           │
           ▼
    Linear(128 → 512)  +  LayerNorm  +  GELU  +  Dropout(0.2)
           │
    Linear(512 → 512)  +  LayerNorm  +  GELU  +  Dropout(0.2)
           │
    Linear(512 → 512)
           │
           ▼
    L2 Normalise
           │
Output: Predicted visual emb  (B, 512)
```

**Loss Function:**

```
L_reg = 0.5 × MSE(pred, GT_visual) + 0.5 × (1 − CosSim(pred, GT_visual))
```
---

### Method 2 — Cross-Modal Distillation (`StudentMLP` + `TeacherProbe`)

A student-teacher framework where the **teacher** is a linear probe trained on frozen VGG16 visual embeddings, and the **student** receives the full AV concatenation.

```
Teacher (frozen after pretraining):
  VGG16 visual (512,)  →  Linear(512, N_classes)  →  class logits

Student:
  Input: AV concat  (B, 640)   ← concat[audio(128); visual(512)]
             │
    Linear(640 → 512) + LayerNorm + GELU + Dropout
             │
    Linear(512 → 512) + LayerNorm + GELU + Dropout
             │
    Linear(512 → 512)  ──►  backbone_emb (B, 512)
             │
    Linear(512 → N_classes)  ──►  student_logits (B, N_classes)
```

**Loss Function (KD with T=4.0, α=0.7, β=0.3):**

```
L_distill = α × T² × KL( softmax(s_logits/T) ‖ softmax(t_logits/T) )
           + β × MSE(s_emb, t_emb)
           + (1−α−β) × (1 − CosSim(s_emb, t_emb))
```

---

### Baseline 1 — Naive Late Fusion (`NaiveLateFusionMLP`)

The minimum viable product. A single-hidden-layer MLP with no normalisation or residual connections — any advanced method must outperform this.

```
Input:  AV concat  (B, 640)
  →  Linear(640 → 512)  →  ReLU
  →  Linear(512 → 512)
  →  L2 Normalise
Output: (B, 512)
```

---

### Baseline 2 — ImageBind / AudioCLIP (State-of-the-Art Benchmark)

Used as a **zero-shot upper bound**. ImageBind naturally aligns image + audio into a joint embedding space without task-specific training. The custom models are benchmarked against this ceiling.

```
ImageBind approximation:
  Audio  (128,)  →  Linear(128, 512)  ─┐
                                        ├─ Concat(1024) → Linear(1024, 512) → L2-norm
  Visual (512,)  →  Linear(512, 512)  ─┘

AudioCLIP:
  Joint audio-visual embedding with CLIP-style contrastive pre-training
```

> **Note:** Full ImageBind weights are not re-trained. The zero-shot projection is used purely as a benchmark ceiling to contextualise the gap between a lightweight custom model and a billion-parameter foundation model.

---

## Evaluation Framework

All methods are evaluated across three primary dimensions defined in the research specification:

### A. Latent Space Proximity

| Metric | Direction |
|--------|-----------|
| **Cosine Similarity** | ↑ Higher is better |
| **MSE** | ↓ Lower is better |

### B. Downstream Retrieval (R@K, MedR)

For each test query, all GT embeddings are ranked by cosine similarity to the predicted embedding. The correct item should appear high in the ranking.

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **R@1** | % of queries where correct item is rank 1 | ↑ Higher is better |
| **R@5** | % of queries where correct item is in top 5 | ↑ Higher is better |
| **R@10** | % of queries where correct item is in top 10 | ↑ Higher is better |
| **MedR** | Median rank of correct item across all queries | ↓ Lower is better |

### C. Computational Efficiency

| Metric | Measurement | Tool |
|--------|------------|------|
| **Inference Latency** | ms / sample (200 runs, 20 warmup, batch=1) | `time.perf_counter` |
| **Parameter Count** | Total trainable parameters | `sum(p.numel())` |
| **FLOPs / MACs** | Multiply-accumulate operations per forward pass | `thop` library |

---

## Results & Benchmarks

> Results change for different datasets, configuration, sample size and training params. Re-run the notebook on the full dataset for final figures. All outputs are saved to `./outputs/`.

---

## Visualisations

The pipeline produces the following outputs automatically in `./outputs/`:

### Dimensionality Reduction Grid

A Matplotlib figure showing how audio, visual, and joint AV embeddings cluster relative to ground-truth video embeddings.

| Method | Input | Output File |
|--------|-------|-------------|
| PCA | AV concat (640,) | `pca_2d_train.png`, `pca_scree.png` |
| t-SNE | PCA-reduced (50d) | `tsne_2d_train.png` |
| UMAP 2D | AV concat | `umap_2d_train.png` |
| UMAP 3D | AV concat | `umap_3d_train.html` *(interactive Plotly)* |

### Training Curves

Loss, MSE, and Cosine Similarity across epochs for both methods:

```
outputs/
  regression_training_curves.png       ← FusionMLP training history
  distillation_training_curves.png     ← StudentMLP training history
  naive_late_fusion.png      ← NLF training history
```

### Final Comparative Dashboard

A 3-row Matplotlib dashboard comparing all methods:

```
Row 1: CosSim | MSE | Recall@K bar chart | MedR
Row 2: Latency | Parameter count | FLOPs | Summary table
Row 3: CosSim distribution violin/box plots 
```

Saved to: `outputs/comparative_dashboard.png`

### Interactive HTML Plots

| File | Description |
|------|-------------|
| `umap_3d.html` | 3D UMAP coloured by class, hover labels |

---

## Repository Structure

```
avavs-dtu/
│
├──  AVAVSS_pipeline_.ipynb        ← Main pipeline notebook (run this)
│
├──  README.md                    ← This file
│
├──  outputs/                     ← Auto-generated on run
│   ├── audioclip_baseline_curves.png
│   ├── audioclip_history.json
│   ├── audioclip_model.pt
│   ├── audioclip_model_final.pt
│   ├── comparative_dashboard.png   ← Main results figure
│   ├── distill_model.pt
│   ├── distillation_curves.png
│   ├── distillation_model_final.pt
│   ├── efficiency_results.json
│   ├── efficiency_table.csv
│   ├── embedding_vis.png
│   ├── full_comparative_results.json
│   ├── naive_baseline_model.pt
│   ├── naive_baseline_model_final.pt
│   ├── naive_history.json
│   ├── naive_late_fusion_baseline_curves.png
│   ├── reg_model.pt
│   ├── regression_curves.png
│   ├── regression_model_final.pt
│   └── umap_3d.html                ← Interactive 3D plot
│
├──  avmit_cache/                 ← Auto-generated on first parse
│   └── avmit_embeddings.npz         ← Cached parsed embeddings
│
├──  report/                      ← Technical report (deliverable)
│   └── Audio-Visual-Approximation-of-Video-Semantic-Space.pdf
│
└──  requirements.txt              ← Python dependencies
---

##  Quick Start

### Option A — Google Colab (Recommended)

Click the badge to open directly in Colab with GPU runtime:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/silver0cloud/avavs-dtu/blob/main/AVAVSS_pipeline.ipynb)

**Steps in Colab:**
1. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
2. Run **Cell 1** — installs all dependencies
3. Run **Cell 5** — downloads the AVMIT dataset from Google Drive
4. Run **Cell 6** — extracts the `.tar` archive
5. Run **all remaining cells** — full pipeline executes end-to-end

> **Estimated runtime:** ~It takes 37 min on T4 GPU for full dataset · it will take ~5 min on a smaller subset (`MAX_SAMPLES=2000`)

---

### Option B — Local Setup

#### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended, not required)
- ~4 GB free disk space for dataset + cache

#### Installation

```bash
# 1. Clone the repository
git clone https://github.com/silver0cloud/avavs-dtu.git
cd avavs-dtu 

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook AVAVSS_pipeline.ipynb
```

#### `requirements.txt`

```text
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
tensorflow>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
umap-learn>=0.5.3
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
open-clip-torch>=2.20.0
gdown>=4.7.0
thop>=0.1.1
tqdm>=4.65.0
h5py>=3.9.0
```

---

## Configuration

All hyperparameters are centralised in the `Config` dataclass. The most important settings for the code:

```python
@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────
    TAR_PATH   : str   = '/content/AVMIT_VGGish_VGG16.tar'
    CACHE_DIR  : str   = './avmit_cache'
    OUTPUT_DIR : str   = './outputs'

    # ── Dataset ──────────────────────────────────────────────
    MAX_SAMPLES   : int   = -1       # -1 = all samples; 2000 for quick test
    VAL_FRACTION  : float = 0.15
    TEST_FRACTION : float = 0.10

    # ── Embedding Dims (fixed by AVMIT pre-extraction) ───────
    AUDIO_DIM  : int = 128           # VGGish output
    VISUAL_DIM : int = 512           # VGG16 fc output
    PROJ_DIM   : int = 256           # audio projection dim
    LATENT_DIM : int = 512           # shared latent space

    # ── Training ─────────────────────────────────────────────
    BATCH_SIZE    : int   = 256
    REG_EPOCHS    : int   = 50       # Latent Space Regression
    DIST_EPOCHS   : int   = 60       # Cross-Modal Distillation
    LR            : float = 3e-4
    WEIGHT_DECAY  : float = 1e-4
    GRAD_CLIP     : float = 1.0
    USE_AMP       : bool  = True     # Mixed precision (GPU only)

    # ── Distillation ─────────────────────────────────────────
    TEMPERATURE   : float = 4.0      # KD soft-target temperature
    ALPHA         : float = 0.7      # KL divergence weight
    BETA          : float = 0.3      # MSE weight

    # ── Evaluation ───────────────────────────────────────────
    RETRIEVAL_K   : list  = [1, 5, 10]
    SEED          : int   = 42
```

---

## Technical Report

The full technical report covering methodology, results, interpretation, and analysis is available here:

> **[Technical Report (PDF)](report/Audio-Visual-Approximation-of-Video-Semantic-Space.pdf)**

### Report Contents

| Section | Coverage |
|---------|----------|
| Abstract | Quick overview of the project |
| 1. Introduction | Problem statement, research question, motives |
| 2. Dataset | AVMIT description, TFRecord parsing, embedding schema |
| 3. Training | LSR, CMD architectures, params, training details |
| 4. Experiments and Evaluation | MSE Loss, CosineSim, Recall@K, MedR |
| Figures | Tables and figures for all three evaluation dimensions |
| Baselines | Naive-Late Fusion, ImageBind, CLIP, AudioCLIP |
| 5. Conclusion | Summary of findings, practical implications |
| 6. References | All cited papers |

---

## Model Weights

All trained model weights are stored on this Google Drive:

> **[ Model Weights (Google Drive)](https://drive.google.com/drive/folders/1CZ2SpHGIDa26_SK5awyYsEBcScYliygq)**

## References

| # | Paper | Conference | Link |
|---|-------|-------|------|
| 1 | Gao et al. — *Listen to Look: Action Recognition by Previewing Audio* | CVPR 2020 | [arXiv](https://arxiv.org/abs/2002.08665) |
| 2 | Girdhar et al. — *ImageBind: One Embedding Space To Bind Them All* | CVPR 2023 | [arXiv](https://arxiv.org/abs/2305.05665) |
| 3 | Guzhov et al. — *AudioCLIP: Extending CLIP to Image, Text and Audio* | ICASSP 2022 | [arXiv](https://arxiv.org/abs/2106.13043) |
| 4 | Shvetsova et al. — *Everything at Once — Multi-modal Fusion Transformer* | CVPR 2022 | [arXiv](https://arxiv.org/abs/2112.04446) |
| 5 | Hinton et al. — *Distilling the Knowledge in a Neural Network* | NeurIPS Workshop 2014 | [arXiv](https://arxiv.org/abs/1503.02531) |
| 6 | Arandjelovic & Zisserman — *Look, Listen and Learn* | ICCV 2017 | [arXiv](https://arxiv.org/abs/1705.08168) |
| 7 | Chen et al. — *CLIP: Learning Transferable Visual Models From Natural Language* | ICML 2021 | [arXiv](https://arxiv.org/abs/2103.00020) |
| 8 | McInnes et al. — *UMAP: Uniform Manifold Approximation and Projection* | JOSS 2018 | [arXiv](https://arxiv.org/abs/1802.03426) |
| 9 | Simonyan & Zisserman — *Very Deep Convolutional Networks (VGG)* | ICLR 2015 | [arXiv](https://arxiv.org/abs/1409.1556) |
| 10 | Hershey et al. — *CNN Architectures for Large-Scale Audio Classification (VGGish)* | ICASSP 2017 | [arXiv](https://arxiv.org/abs/1609.09430) |

---

## Acknowledgements

The AVMIT dataset pre-extracted features (VGGish + VGG16) are used under their respective research licenses. 
ImageBind and AudioCLIP are used in evaluation-only zero-shot mode.

---

<div align="center">

**for AIMS DTU Research Internship 2026**

*If you use this pipeline and find it helpful, consider starring the repository!*

[![GitHub stars](https://img.shields.io/github/stars/silver0cloud/avavs-dtu?style=social)](https://github.com/silver0cloud/avavs-dtu)

</div>

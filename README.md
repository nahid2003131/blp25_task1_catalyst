# BLP-2025 Task 1 — Catalyst
### Transformer Ensembles and Multi-task Learning Approaches for Bangla Hate Speech Detection

This repository contains Colab-ready notebooks, model configurations, and analysis scripts for the system **“Catalyst at BLP-2025 Task 1: Transformer Ensembles and Multi-task Learning Approaches for Bangla Hate Speech Detection”**, accepted to the **Second International Workshop on Bangla Language Processing (BLP-2025)**.

---

## 📄 Overview

The project targets **Bangla Hate Speech Identification** under the BLP-2025 shared task, consisting of three subtasks:

| Subtask | Description |
|----------|-------------|
| **1A** | Hate Type Classification — six-way classification of the hate category (*Abusive, Sexism, Religious Hate, Political Hate, Profane, None*). |
| **1B** | Hate Target Identification — predicts the target group of hate (*Individual, Organization, Community, Society,* or *None*). |
| **1C** | Multi-task Classification — joint prediction of hate **Type**, **Target**, and **Severity** using a shared encoder with task-specific heads. |

Our approach combines **four multilingual Transformer encoders** and applies:
- **Hard-voting ensembles** for Subtasks 1A and 1B  
- **Shared-encoder multi-task learning** for Subtask 1C  

---

## 🧩 Model Components

| Encoder | Source | Notes |
|----------|---------|-------|
| `XLM-RoBERTa-large` | [Hugging Face](https://huggingface.co/xlm-roberta-large) | Strong cross-lingual coverage |
| `mDeBERTa-v3-base` | [Hugging Face](https://huggingface.co/microsoft/mdeberta-v3-base) | Efficient disentangled attention |
| `MuRIL-base-cased` | [Hugging Face](https://huggingface.co/google/muril-base-cased) | Optimized for Indian languages |
| `IndicBERTv2-MLM-only` | [AI4Bharat](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) | Lightweight, Indic-specific encoder |

---

## ⚙️ Repository Structure

```
blp25_task1_catalyst/
│
├── subtask_1A/             # Colab notebooks for Hate Type Classification
├── subtask_1B/             # Colab notebooks for Target Identification
├── subtask_1C/             # Colab notebooks for Multi-task Classification
│
├── assembler.ipynb         # Hard-voting ensemble and submission generation
├── error_analysis.ipynb    # Quantitative & qualitative error analysis
├── README.md               # Project documentation
```

---

## 🚀 Running on Google Colab

### 1️⃣ Open in Colab
Each subtask folder contains a `.ipynb` notebook that can be opened directly in Google Colab.  
Click **“Open in Colab”** on the top of each notebook or upload manually.

### 2️⃣ Install Dependencies
Run the first cell in each notebook to install the required libraries:

```python
!pip install -r requirements.txt
```

### 3️⃣ Training & Inference
Run the notebook cells sequentially to:
- Load the dataset  
- Fine-tune transformer models  
- Evaluate using official metrics  
- Save predictions to `/outputs`

Example:
```python
# Inside Colab
from google.colab import drive
drive.mount('/content/drive')
# Then execute training cells as provided
```

### 4️⃣ Ensemble Generation
After generating individual predictions for 1A and 1B, run:
```bash
# From Colab or local runtime
!jupyter nbconvert --to notebook --execute assembler.ipynb
```

### 5️⃣ Multi-task Model
The `subtask_1C/` notebook trains a shared-encoder model (IndicBERTv2) with multi-head outputs for hate type, target, and severity.

---

## 📊 Results Summary (Official Test Scores)

| Subtask | Approach | Micro-F1 | Leaderboard Rank |
|----------|-----------|-----------|------------------|
| **1A** | Hard-voting Ensemble (4 models) | **73.05** | 7 / 37 |
| **1B** | Hard-voting Ensemble (3 models) | **72.79** | 8 / 24 |
| **1C** | Shared Encoder (IndicBERTv2) | **72.40** | 10 / 21 |

---

## 🔍 Error Analysis Highlights
- **Type Confusion:** Abusive ↔ Political Hate overlap (448 cases)  
- **Severity Calibration:** ~26 % confusion between *Mild* and *None*  
- **Target Ambiguity:** Frequent overlap among *Organization*, *Individual*, and *Community*  

Detailed confusion matrices and error samples are available in `error_analysis.ipynb`.

---

## 💡 Future Work
- Class-balanced or focal loss for imbalance mitigation  
- Calibrated soft-voting ensembles and ordinal severity modeling  
- Domain-adaptive pretraining on unlabeled Bangla social media text  

---

## 🏆 Acknowledgments
This work was conducted as part of the **BLP-2025 Shared Task on Bangla Hate Speech Identification**, organized by the **Bangla Language Processing Workshop**.  
We thank the task organizers and reviewers for their constructive feedback.

---

© 2025. Licensed under the [MIT License](LICENSE).

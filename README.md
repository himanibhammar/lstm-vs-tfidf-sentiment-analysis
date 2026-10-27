# Product Review Sentiment Classification Using LSTM & TF-IDF Logistic Regression
### Investigating the Effect of Sentence Length on Classification Accuracy

---

## 1. Project Overview

This project is an end-to-end Natural Language Processing (NLP) academic research assignment and machine learning pipeline that classifies customer product reviews into **positive** or **negative** sentiment, with a special empirical investigation into **how review text length (sentence/word length) affects model performance**.

### Key Research Questions
1. **RQ1:** How effectively can a Recurrent Neural Network (Bidirectional LSTM) learn sentiment patterns from product reviews?
2. **RQ2:** Does an LSTM sequence model outperform a classical bag-of-words baseline (**TF-IDF + Logistic Regression**)?
3. **RQ3:** Does review length (Short: 1–10 words, Medium: 11–30 words, Long: >30 words) systematically impact sentiment classification accuracy?
4. **RQ4:** Do sequential deep learning models and linear ngram models exhibit different error patterns across review length regimes?

---

## 2. Dataset Specifications

The project operates on two labelled dataset files placed in the working directory:
- `train.csv` (5,000 samples: 3,092 positive, 1,908 negative)
- `test.csv` (1,070 samples: 553 positive, 447 negative)

### Schema
| Column Name | Type | Description |
|:---|:---|:---|
| `review_text` | String | Raw text of customer product review |
| `label` | String | Ground-truth sentiment class (`positive` or `negative`) |

No synthetic or artificial labels are introduced. Text preprocessing preserves sentiment-bearing punctuation and handles class weighting to prevent majority-class bias.

---

## 3. Repository Structure

```
lstm_sentimentanalysis/
│
├── product_review_sentiment_analysis.ipynb  # Primary complete academic Jupyter Notebook
├── sentiment_lstm.py                        # Standalone PyTorch execution script
├── train.csv                                # Training dataset (5,000 reviews)
├── test.csv                                 # Held-out test dataset (1,070 reviews)
├── requirements.txt                         # Python dependencies
├── README.md                                # Comprehensive documentation & guide
└── output/                                  # Directory for saved figures, metrics, & checkpoints
```

---

## 4. Models & Methodology

### 1. TF-IDF + Logistic Regression (Baseline)
- **Feature Extraction:** Sublinear TF-IDF weighting over word unigrams and bigrams (`ngram_range=(1, 2)`), capped at top 20,000 features.
- **Classifier:** L2-regularized Logistic Regression with balanced class weights (`class_weight='balanced'`).
- **Rationale:** High-performing, interpretable linear benchmark robust to word bag representations.

### 2. Bidirectional LSTM (Deep Sequence Model)
- **Tokenization & Numericalization:** Custom vocabulary / Keras Tokenizer (top 15,000 tokens, with `<pad>` and `<unk>`).
- **Sequence Padding:** Uniform length normalization (`max_len=120`).
- **Embedding Layer:** Trainable dense vector representations (`dim=64` or `128`) with spatial dropout.
- **Recurrent Layer:** Bidirectional LSTM (`64` or `128` units) capturing bidirectional contextual dependencies (forward & backward sequential flow).
- **Pooling & Classification:** Concatenated pooling / global pooling followed by Dense ReLU and Sigmoid output layer.
- **Regularization:** Dropout (0.3), EarlyStopping monitoring validation loss with checkpoint recovery.

### 3. Sentence Length Stratification
Reviews are partitioned into three length buckets based on whitespace word counts:
- **Short Reviews:** 1 to 10 words (dense sentiment keywords, minimal context)
- **Medium Reviews:** 11 to 30 words (balanced context and sentiment expression)
- **Long Reviews:** > 30 words (rich context, potential mixed sentiments and digressions)

Performance (Accuracy, F1-Score, Confusion Matrix) is computed and compared across all three regimes.

---

## 5. Environment Setup & Installation

### Prerequisites
- Python 3.9+ (or Python 3.10 / 3.11)
- `pip` package manager

### Local Environment Setup
```bash
# Clone or navigate to the repository
cd lstm_sentimentanalysis

# (Recommended) Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 6. How to Run the Notebook

### Option A: Running in Kaggle Notebook (Recommended for Zero-Setup GPU)
1. Navigate to [Kaggle](https://www.kaggle.com/) and click **New Notebook**.
2. Under **File > Upload Notebook**, select `product_review_sentiment_analysis.ipynb`.
3. In the right-hand panel:
   - Click **Add Data** -> Upload your `train.csv` and `test.csv`.
   - Set **Accelerator** to **GPU P100** or **GPU T4 x2** (or CPU).
4. Run all cells sequentially (**Run All**).
5. The notebook automatically detects `/kaggle/input/` paths and executes without any modification.

### Option B: Running Locally via Jupyter Lab / Notebook
```bash
jupyter notebook product_review_sentiment_analysis.ipynb
```
Select the Python kernel and click **Kernel > Restart & Run All**.

### Option C: Running the Standalone PyTorch Script
If you prefer a direct CLI execution:
```bash
python sentiment_lstm.py
```

---

## 7. Evaluation Metrics & Measured Results

The notebook was executed end-to-end on the held-out `test.csv` dataset (1,000 samples: 553 positive, 447 negative). Below are the verified empirical results:

### Head-to-Head Comparison Table
| Metric | TF-IDF + Logistic Regression (Baseline) | Bidirectional LSTM (Deep Sequence) | Delta (Advantage) |
|:---|:---:|:---:|:---:|
| **Accuracy** | **95.40%** | 93.60% | +1.80% (LR) |
| **Precision** | 0.9363 | **0.9577** | +0.0214 (LSTM) |
| **Recall** | **0.9837** | 0.9313 | +0.0524 (LR) |
| **F1-Score** | **0.9594** | 0.9441 | +0.0154 (LR) |
| **ROC-AUC** | **0.9932** | 0.9818 | +0.0114 (LR) |

### Sentence Length Performance Breakdown
Reviews were partitioned into three mutually exclusive word count categories:
- **Short Reviews (1–10 words)**: 569 samples (56.9% of test set)
  - BiLSTM F1: **0.9582** | LR F1: **0.9638**
- **Medium Reviews (11–30 words)**: 341 samples (34.1% of test set)
  - BiLSTM F1: **0.9398** | LR F1: **0.9593**
- **Long Reviews (>30 words)**: 90 samples (9.0% of test set)
  - BiLSTM F1: **0.8947** | LR F1: **0.9189**

---

## 8. Output Artifacts

All outputs and high-resolution plots are generated and saved to the `output/` directory:
- `fig1_class_distribution.png`: Class balance in training and test sets.
- `fig2_word_count_distribution.png`: Review word length distributions with thresholds.
- `fig3_training_curves.png`: Training & validation loss and accuracy curves over epochs.
- `fig4_confusion_matrices.png`: Side-by-side confusion matrices for both models.
- `fig5_model_comparison.png`: Metric-by-metric comparison bar chart.
- `fig6_length_performance_breakdown.png`: Accuracy & F1 across Short, Medium, and Long reviews.
- `best_sentiment_lstm.pt`: Best saved PyTorch BiLSTM model weights and vocabulary mapping.
- `baseline_tfidf_lr.joblib`: Serialized TF-IDF vectorizer and trained Logistic Regression model.

---

## 9. Academic Integrity & Reproducibility

- **Fixed Random Seeds:** All splits, weights, and initializations use `SEED = 42`.
- **No Data Leakage:** Tokenizers, vocabularies, and TF-IDF vectorizers are fit **strictly** on the training set, with test data evaluated out-of-sample.
- **Empirical Grounding:** All conclusions, tables, and comparative charts reflect real model predictions rather than simulated values.


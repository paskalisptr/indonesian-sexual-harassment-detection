# Indonesian Sexual Harassment Detection on Social Media X

A text classification project to detect sexual harassment (SH) content in informal Indonesian tweets, comparing three approaches: a classical model, a sequential deep learning model, and an IndoBERT-based transformer.

> Group project (2 members), completed for the *Simulation Modelling and Analysis* course. Personal contribution: **the entire modeling pipeline** (TF-IDF+LR baseline, BiLSTM-GRU, and IndoBERT fine-tuning), including evaluation and result analysis.

---

## Background

Sexual harassment on social media (particularly Twitter/X) is rising in Indonesia. Komnas Perempuan (Indonesia's National Commission on Violence Against Women) recorded a 35% increase in reported cases in 2024, and viral cases have triggered hundreds of thousands of mentions on X within a short time. However, automated detection systems for Indonesian remain limited, especially for handling informal text full of slang and code-switching.

This project builds and compares three classification models to detect SH content from Indonesian tweet text.

## Dataset

- **Source:** Tweets from X/Twitter, collected using Tweet-Harvest (Jan–Feb 2026, ~55 unique keywords)
- **Size:** 1,608 tweets after preprocessing & deduplication
  - Non-SH: 1,221 (75.6%)
  - SH: 387 (24.4%)
  - Imbalance ratio: 3.18:1

### Preprocessing (Semantic-Preserving Pipeline)
The preprocessing pipeline was designed to preserve semantic meaning (no aggressive stemming), including: case folding, URL removal, mention anonymization (`@username` → generic token), emoji-to-descriptive-token conversion, character elongation normalization (`soooo` → `soo`), and tokenization with a negation whitelist (`tidak`, `bukan`, `jangan`, Indonesian negation words) so that negation meaning isn't lost during stopword removal.

## Methods

Three models were built and compared head-to-head:

| Model | Type | Notes |
|---|---|---|
| TF-IDF + Logistic Regression | Classical baseline | Optimized with SMOTE, GridSearchCV, Stratified K-Fold |
| BiLSTM-GRU | Sequential deep learning | Captures contextual word dependencies |
| IndoBERT (fine-tuned) | Transformer | `indobenchmark/indobert-base-p2`, fine-tuned for binary classification |

**Additional pipeline:** Chi-Square & RFE for feature selection, SMOTE for class imbalance handling, TruncatedSVD/t-SNE for dimensionality reduction exploration.

**Tools:** Python, PyTorch, scikit-learn, imbalanced-learn (imblearn), Hugging Face Transformers

## Results

| Model | F1-Macro | ROC-AUC | Training Time |
|---|---|---|---|
| TF-IDF + LR | 0.7539 | 0.8525 | 0.04s |
| BiLSTM-GRU | 0.7040 | 0.7765 | 101.3s |
| **IndoBERT** | **0.7902** | **0.8744** | 41.4s |

IndoBERT delivered the best performance with far better training efficiency than BiLSTM-GRU, thanks to the Indonesian language knowledge already learned during pre-training. The model reached optimal performance after just one epoch; further fine-tuning led to catastrophic forgetting.

### Other Analytical Findings
- 23.5% of scraping keywords showed lexical bias (IoU > 0.5), indicating potential sampling bias from the keyword-based scraping approach
- Chi-Square-based feature selection actually decreased TF-IDF performance on short text. PCA/TruncatedSVD proved ineffective for sparse representations in this case
- All three models consistently produced more false negatives than false positives, tending to underpredict the minority class (SH). This is an important consideration for this use case, since missing real cases is riskier than a false alarm

## Limitations

- Dataset size (N=1,608) is relatively small for sequential deep learning models
- Vulgar-keyword-based scraping may miss SH tweets that don't use explicit vulgar language
- The model was only validated on the X platform, not yet tested on other platforms (Instagram, TikTok)
- The definition of sexual harassment is cultural and contextual, so the model would need adaptation for different regional/community contexts

## Future Work

- Expand the dataset (targeting 5,000 samples per class) through data augmentation
- Explore IndoBERTweet (a model specifically pre-trained on Indonesian Twitter data)
- Multi-source scraping strategy to reduce lexical bias
- Cross-platform validation and concept drift analysis over time

## Repository Structure

```
├── deteksi_kekerasan_seksual.ipynb   # preprocessing, TF-IDF+LR, BiLSTM-GRU, IndoBERT (end-to-end)
├── dataset_preprocessed.csv
└── README.md
```

The entire pipeline (preprocessing → three models → evaluation) is contained in a single notebook, organized with section headings (e.g. `## 1. Preprocessing`, `## 2. Baseline: TF-IDF + LR`, etc.) for easy navigation.

## References

- Koto, F., et al. (2020). IndoLEM and IndoBERT. *COLING*.
- Koto, F., Lau, J. H., & Baldwin, T. (2021). IndoBERTweet. *IALP*.
- Chawla, N. V., et al. (2002). SMOTE. *JAIR*, 16.
- Satria, H. (2023). Tweet-Harvest. GitHub.

# Deteksi Kekerasan Seksual Berbahasa Indonesia di Media Sosial X

Proyek klasifikasi teks untuk mendeteksi konten kekerasan seksual (KS) dalam tweet berbahasa Indonesia informal, membandingkan tiga pendekatan: model klasik, deep learning sekuensial, dan transformer berbasis IndoBERT.

> Proyek kelompok (2 anggota) — dikerjakan untuk mata kuliah *Simulation Modelling and Analysis*. Kontribusi personal pada proyek ini: **seluruh pipeline modeling** (baseline TF-IDF+LR, BiLSTM-GRU, dan fine-tuning IndoBERT), termasuk evaluasi dan analisis hasil.

---

## Latar Belakang

Pelecehan seksual melalui media sosial (khususnya Twitter/X) meningkat di Indonesia — data Komnas Perempuan mencatat kenaikan 35% kasus pada 2024, dan kasus-kasus viral memicu ratusan ribu mention di X dalam waktu singkat. Namun, sistem deteksi otomatis untuk bahasa Indonesia masih terbatas, terutama untuk menangani teks informal yang penuh slang dan *code-switching*.

Proyek ini membangun dan membandingkan tiga model klasifikasi untuk mendeteksi konten KS dari teks tweet bahasa Indonesia.

## Dataset

- **Sumber:** Tweet dari X/Twitter, dikumpulkan menggunakan Tweet-Harvest (Jan–Feb 2026, ~55 kata kunci unik)
- **Ukuran:** 1.608 tweet setelah preprocessing & deduplication
  - Non-KS: 1.221 (75,6%)
  - KS: 387 (24,4%)
  - *Imbalance ratio* 3,18:1

### Preprocessing (Semantic-Preserving Pipeline)
Pipeline preprocessing dirancang untuk mempertahankan makna semantik teks (tidak menggunakan stemming agresif), meliputi: case folding, penghapusan URL, anonimisasi mention (`@user` → token generik), konversi emoji ke token deskriptif, normalisasi elongasi karakter (`sangeeeee` → `sangee`), serta tokenisasi dengan whitelist negasi (`tidak`, `bukan`, `jangan`) agar makna penyangkalan tidak hilang saat stopword removal.

## Metode

Tiga model dibangun dan dibandingkan secara head-to-head:

| Model | Tipe | Catatan |
|---|---|---|
| TF-IDF + Logistic Regression | Baseline klasik | Dioptimasi dengan SMOTE, GridSearchCV, Stratified K-Fold |
| BiLSTM-GRU | Deep learning sekuensial | Menangkap dependensi kontekstual antar kata |
| IndoBERT (fine-tuned) | Transformer | `indobenchmark/indobert-base-p2`, fine-tuned untuk klasifikasi biner |

**Pipeline tambahan:** Chi-Square & RFE untuk feature selection, SMOTE untuk menangani class imbalance, TruncatedSVD/t-SNE untuk eksplorasi dimensionality reduction.

**Tools:** Python, PyTorch, scikit-learn, imbalanced-learn (imblearn), Hugging Face Transformers

## Hasil

| Model | F1-Macro | ROC-AUC | Waktu Training |
|---|---|---|---|
| TF-IDF + LR | 0.7539 | 0.8525 | 0.04s |
| BiLSTM-GRU | 0.7040 | 0.7765 | 101.3s |
| **IndoBERT** | **0.7902** | **0.8744** | 41.4s |

IndoBERT memberikan performa terbaik dengan waktu training jauh lebih efisien dibanding BiLSTM-GRU, berkat pengetahuan bahasa Indonesia yang sudah dipelajari saat pre-training. Model ini mencapai performa optimal hanya di epoch pertama — fine-tuning lebih lanjut menyebabkan *catastrophic forgetting*.

### Temuan Analitis Lain
- 23,5% keyword scraping menunjukkan *lexical bias* (IoU > 0,5), mengindikasikan potensi sampling bias dari pendekatan keyword-based scraping
- *Feature selection* berbasis Chi-Square justru menurunkan performa TF-IDF pada teks pendek — PCA/TruncatedSVD tidak efektif untuk representasi sparse di kasus ini
- Ketiga model konsisten menghasilkan lebih banyak *false negative* daripada *false positive* — cenderung *underpredict* kelas minoritas (KS), pertimbangan penting untuk use-case ini karena melewatkan kasus nyata lebih berisiko daripada false alarm

## Limitasi

- Ukuran dataset (N=1.608) relatif kecil untuk model deep learning sekuensial
- Scraping berbasis keyword vulgar berpotensi melewatkan tweet KS yang tidak menggunakan kata vulgar eksplisit
- Model hanya divalidasi pada platform X, belum diuji di platform lain (Instagram, TikTok)
- Definisi kekerasan seksual bersifat kultural dan kontekstual — model memerlukan adaptasi untuk konteks daerah/komunitas berbeda

## Future Work

- Memperluas dataset (target 5.000 sampel per kelas) melalui augmentasi data
- Eksplorasi IndoBERTweet (model yang dilatih khusus pada data Twitter Indonesia)
- Strategi scraping multi-sumber untuk mengurangi lexical bias
- Validasi lintas platform dan analisis *concept drift* dari waktu ke waktu

## Struktur Repository

```
├── deteksi_kekerasan_seksual.ipynb   # preprocessing, TF-IDF+LR, BiLSTM-GRU, IndoBERT — end-to-end
├── data/                              # (atau .gitignore jika berisi data sensitif)
└── README.md
```

> ⚠️ Ganti nama file notebook di atas dengan nama file aslimu sebelum upload.

Seluruh pipeline (preprocessing → tiga model → evaluasi) ada dalam satu notebook. Gunakan heading/section di dalam notebook (misal `## 1. Preprocessing`, `## 2. Baseline: TF-IDF + LR`, dst) supaya recruiter yang membuka notebook bisa langsung navigasi tanpa scroll bingung — kalau notebookmu belum punya heading section yang jelas, ini worth ditambahkan sebelum upload.

## Referensi

- Koto, F., et al. (2020). IndoLEM and IndoBERT. *COLING*.
- Koto, F., Lau, J. H., & Baldwin, T. (2021). IndoBERTweet. *IALP*.
- Chawla, N. V., et al. (2002). SMOTE. *JAIR*, 16.
- Satria, H. (2023). Tweet-Harvest. GitHub.

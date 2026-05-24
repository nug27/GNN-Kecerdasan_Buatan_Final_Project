# 🛒 SecondPrice: Prediksi Harga Jual Barang Bekas dengan Graph Neural Network

**Mata Kuliah:** Kecerdasan Buatan — Teknik Komputer, Universitas Indonesia  
**Model Utama:** Graph Neural Network (GraphSAGE / GAT)  
**Dataset:** Mercari Price Suggestion Dataset (Kaggle)

---

## 📌 Latar Belakang

Pasar jual beli barang bekas di Indonesia terus berkembang pesat, didorong oleh platform seperti OLX, Carousell, dan Facebook Marketplace. Salah satu tantangan terbesar yang dihadapi penjual adalah menentukan harga jual yang wajar — terlalu mahal membuat barang tidak laku, terlalu murah merugikan penjual. Saat ini, sebagian besar penjual menentukan harga hanya berdasarkan intuisi atau perbandingan manual yang memakan waktu.

Proyek ini hadir untuk menjawab tantangan tersebut dengan membangun sistem prediksi harga otomatis berbasis **Graph Neural Network (GNN)** yang mampu memahami konteks pasar secara holistik — bukan hanya melihat fitur produk secara individual, tetapi juga relasi antar produk, merek, dan kategori di dalam ekosistem pasar.

---

## 🎯 Tujuan Proyek

Membangun model GNN yang dapat memprediksi harga jual wajar suatu barang bekas berdasarkan input berupa nama produk, merek, kategori, kondisi barang, dan deskripsi singkat — tanpa memerlukan foto.

---

## 🧠 Pendekatan: Graph Neural Network

Ide utama proyek ini adalah merepresentasikan data pasar sebagai sebuah **graph**, di mana:

- **Node** merepresentasikan entitas: produk, merek, dan kategori.
- **Edge** merepresentasikan relasi: "produk X termasuk merek Y", "produk X masuk kategori Z", "produk X dan Y sering berada di segmen harga yang sama".

Dengan struktur graph ini, GNN dapat melakukan *message passing* — setiap node mempelajari informasi dari tetangga-tetangganya. Hasilnya, model tidak hanya melihat fitur satu produk secara terisolasi, tetapi juga memahami posisi produk tersebut di dalam konteks pasar yang lebih luas. Misalnya, iPhone 12 bekas akan mendapatkan konteks dari produk Apple lainnya, dari kategori Smartphone secara keseluruhan, serta dari produk-produk dengan kondisi serupa.

Arsitektur model yang digunakan adalah **GraphSAGE** atau **Graph Attention Network (GAT)**, dengan output berupa nilai regresi (prediksi harga).

```
[Fitur Produk: nama, kondisi, deskripsi]
              ↓
     Direpresentasikan sebagai Node
              ↓
    Graph dibangun: Produk ↔ Merek ↔ Kategori
              ↓
     GNN (GraphSAGE / GAT) — Message Passing
              ↓
      Regression Head → Prediksi Harga (Rp)
```

---

## 📂 Dataset

Dataset yang digunakan adalah **Mercari Price Suggestion Dataset** dari Kaggle, yang merupakan dataset resmi dari kompetisi machine learning Mercari — platform e-commerce terbesar di Jepang.

- **Link:** [kaggle.com/c/mercari-price-suggestion-challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge)
- **Ukuran:** 1,4 juta+ listing produk
- **Kolom utama:** `name`, `brand_name`, `category_name`, `item_condition_id`, `item_description`, `price`, `shipping`
- **Rencana penggunaan:** Subset 100.000–200.000 baris untuk efisiensi training di Google Colab

---

## 🔬 Rencana Eksperimen

Untuk memenuhi kriteria **Evaluasi Komparatif (+10% bonus)**, proyek ini akan menjalankan tiga eksperimen yang dibandingkan secara sistematis:

| Eksperimen | Model | Deskripsi |
|---|---|---|
| Baseline 1 | MLP / XGBoost | Prediksi dari fitur tabular biasa, tanpa graph |
| Baseline 2 | TF-IDF + Ridge Regression | Pendekatan NLP klasik pada deskripsi teks |
| **Model Utama** | **GNN (GraphSAGE/GAT)** | **Prediksi berbasis struktur graph pasar** |

Hipotesis utama: GNN akan menghasilkan error (RMSE/MAE) lebih rendah dibanding kedua baseline karena mampu memanfaatkan informasi kontekstual dari relasi antar node.

---

## 📊 Metrik Evaluasi

- **MAE** (Mean Absolute Error) — selisih rata-rata harga prediksi vs aktual
- **RMSE** (Root Mean Squared Error) — penalti lebih besar untuk error ekstrem
- **RMSLE** (Root Mean Squared Log Error) — metrik standar Mercari, robust terhadap outlier harga
- **R² Score** — seberapa baik model menjelaskan variansi harga

---

## 🗂️ Pembagian Tugas

| Anggota | Tanggung Jawab |
|---|---|
| **Anggota A** | Preprocessing data, pembangunan struktur graph, node & edge feature engineering |
| **Anggota B** | Implementasi model GNN (GraphSAGE/GAT) dengan PyTorch Geometric, training & tuning |
| **Anggota C** | Implementasi baseline models, evaluasi komparatif, laporan & slide presentasi |

> Seluruh anggota wajib memahami keseluruhan sistem untuk persiapan wawancara teknis.

---

## 🎁 Strategi Bonus (+10%)

| Kriteria Bonus | Strategi |
|---|---|
| ✅ Evaluasi Komparatif | Bandingkan GNN vs MLP vs TF-IDF+Ridge secara terstruktur dengan tabel dan grafik |
| ✅ Analisis Etika & Bias | Analisis bias merek (Apple vs lokal), bias kategori, dan edge cases produk antik/langka |

---

## 🛠️ Tech Stack

| Komponen | Tools |
|---|---|
| Bahasa | Python 3.10+ |
| Framework GNN | PyTorch Geometric |
| NLP / Embedding | Sentence-Transformers / TF-IDF |
| Baseline | XGBoost, Scikit-learn |
| Environment | Google Colab (GPU T4) |
| Demo | Gradio |
| Versioning | GitHub |

---

## 🎬 Demo Plan

Saat sesi demonstrasi, kelompok akan menampilkan antarmuka interaktif berbasis **Gradio** di mana penguji dapat:

1. Menginput nama produk, merek, kategori, dan kondisi secara bebas
2. Melihat prediksi harga secara real-time dari model GNN
3. Membandingkan hasil prediksi GNN vs baseline secara langsung di layar

Contoh input yang akan disiapkan:
```
Nama     : "iPhone 12 64GB"
Merek    : Apple
Kategori : Electronics > Phones > iPhone
Kondisi  : 3 (Good)
Deskripsi: "Minor scratch on the back, battery health 87%"

→ Prediksi GNN  : $185 – $210
→ Prediksi MLP  : $172
→ Harga Aktual  : $195 ✅
```

---

## 📅 Timeline Pengerjaan

| Minggu | Target |
|---|---|
| Minggu 1 | EDA dataset, preprocessing, pembangunan graph |
| Minggu 2 | Training GNN + baseline, hyperparameter tuning |
| Minggu 3 | Evaluasi komparatif, analisis bias, pembuatan slide & demo Gradio |
| **25 Mei 2026** | **Deadline pengumpulan via EMAS** |

---

*Dokumen ini merupakan overview awal proyek dan dapat diperbarui seiring perkembangan pengerjaan.*

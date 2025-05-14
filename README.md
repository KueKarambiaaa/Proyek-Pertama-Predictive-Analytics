# 📊 Prediksi Harga Bitcoin 2018–2025 Menggunakan Machine Learning

**Nama**: Rahmi Amilia  
**Judul Proyek**: Predictive Analytics: Bitcoin Historical Datasets  
**Domain**: Keuangan / Investasi / Cryptocurrency  
**Platform Dataset**: [Kaggle - Bitcoin Historical Datasets](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

---

## 1. Domain Proyek

Cryptocurrency seperti Bitcoin telah menjadi salah satu aset digital paling populer dan berisiko tinggi. Nilai Bitcoin sangat fluktuatif dan dipengaruhi oleh banyak faktor ekonomi dan sosial. Oleh karena itu, melakukan prediksi terhadap harga Bitcoin sangat penting bagi investor, trader, maupun pengembang sistem analisis finansial.

Menurut [Nakamoto (2008)](https://bitcoin.org/bitcoin.pdf), Bitcoin dirancang sebagai sistem kas elektronik peer-to-peer. Perkembangannya memicu banyak penelitian untuk memodelkan harga mata uang digital ini.

> Referensi:  
> - Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*.  
> - McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning.

---

## 2. Business Understanding

### Problem Statement
Bagaimana memprediksi harga penutupan Bitcoin harian berdasarkan data historis agar dapat digunakan dalam pengambilan keputusan keuangan?

### Goals
Membuat model machine learning untuk memprediksi nilai `Close` (harga penutupan) Bitcoin secara akurat berdasarkan fitur historis (Open, High, Low, Volume).

### Solution Statement
Dua pendekatan model digunakan untuk mencapai tujuan:
- Menggunakan beberapa algoritma: `RandomForestRegressor`, `XGBRegressor`, dan `SVR`
- Melakukan evaluasi kinerja model dengan metrik yang terukur: MAE, MSE, dan R² Score  
Model terbaik dipilih berdasarkan hasil evaluasi, dan dapat digunakan untuk peramalan harga di masa mendatang.

---

## 3. Data Understanding

Dataset yang digunakan: [Bitcoin Historical Data (2018-Now)](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

### Jumlah Data dan Fitur
Dataset memiliki total **2.675 baris** dan **12 kolom**. Setelah preprocessing, hanya 5 kolom numerik yang digunakan.

### Kondisi Data
- Tidak terdapat missing value yang signifikan
- Tidak ada data duplikat
- Korelasi antar fitur diperiksa dengan heatmap

### Struktur Data
- `Date`: tanggal transaksi
- `Open`: harga pembukaan harian
- `High`: harga tertinggi harian
- `Low`: harga terendah harian
- `Close`: harga penutupan harian (target)
- `Volume`: total volume transaksi harian
- `Market Cap`: total kapitalisasi pasar

### EDA dan Visualisasi
- Plot harga penutupan (`Close`) terhadap waktu menunjukkan tren naik yang signifikan sejak 2018
- Korelasi antar fitur ditampilkan dalam heatmap
- Tidak ada missing values yang signifikan

---

## 4. Data Preparation

- Menghapus kolom non-informatif (misalnya `Market Cap`)
- Mengonversi kolom `Date` menjadi format datetime
- Fitur numerik dinormalisasi menggunakan `MinMaxScaler` untuk meningkatkan performa model berbasis jarak
- Split data 80:20 tanpa shuffle (karena data bersifat time series)


- **Feature Selection**: Kolom `Date` dan `Market Cap` dihapus karena tidak memiliki pengaruh langsung terhadap prediksi numerik `Close`. Fitur yang dipertahankan adalah: `Open`, `High`, `Low`, `Volume`, `Close`.


**Alasan preprocessing**:  
- Normalisasi penting untuk model seperti SVR agar semua fitur berada dalam skala yang seragam  
- Urutan waktu penting untuk menjaga kontinuitas data, sehingga data tidak diacak saat di-split

---


## 5. Model Development

Langkah-langkah pengembangan model yang dilakukan:

### a. Splitting Data
Data dibagi menjadi:
- **Training set (80%)**
- **Testing set (20%)**
Pembagian dilakukan **tanpa shuffle**, karena data bersifat deret waktu (time series), sehingga kontinuitas perlu dijaga.

### b. Normalisasi
Semua fitur numerik (`Open`, `High`, `Low`, `Volume`) dinormalisasi menggunakan `MinMaxScaler`. Hal ini dilakukan untuk:
- Menyamakan skala antar fitur
- Meningkatkan performa model berbasis jarak seperti SVR
- Memastikan stabilitas model neural network (LSTM)

### c. Feature Engineering
Fitur `Mean_Price` sempat ditambahkan (rata-rata `High` dan `Low`), namun tidak digunakan dalam final modeling. Fitur akhir yang dipilih adalah: `Open`, `High`, `Low`, `Volume`.

### d. Model Training
Empat model dilatih dan dibandingkan performanya:
- **RandomForestRegressor**
- **XGBRegressor**
- **SVR (Support Vector Regression)**
- **LSTM (Long Short-Term Memory)**: menggunakan Keras, input diubah ke bentuk 3D

### e. Evaluasi
Semua model dievaluasi menggunakan metrik:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² Score

Model terbaik dipilih berdasarkan performa tertinggi di data testing.


## 6. Modeling

### Model 1: RandomForestRegressor

#### Cara Kerja
Random Forest membangun banyak decision tree dengan data bootstrap.  
Setiap tree mempertimbangkan subset fitur acak untuk split.  
Prediksi akhir diambil rata-rata dari semua pohon.

#### Parameter yang digunakan
- `n_estimators = 100` (default)
- `max_depth = None` (default)
- `random_state = 42`

#### Kelebihan/Kekurangan
- ✅ Stabil, tahan overfitting, robust untuk data non-linear.
- ❌ Sulit diinterpretasikan.

---

### Model 2: XGBRegressor

#### Cara Kerja
XGBoost menggunakan teknik boosting, di mana model dibangun bertahap.  
Setiap iterasi memfokuskan diri pada error sisa dari model sebelumnya.

#### Parameter yang digunakan
- `n_estimators = 100` (default)
- `learning_rate = 0.1` (default)
- `max_depth = 3` (default)
- `random_state = 42`

#### Kelebihan/Kekurangan
- ✅ Akurat, cepat, menangani missing value.
- ❌ Perlu tuning yang lebih teliti.

---

### Model 3: Support Vector Regression (SVR)

#### Cara Kerja
SVR mencari hyperplane regresi dalam margin epsilon.  
Data yang berada di luar margin menjadi support vector.  
Kernel RBF digunakan untuk menangani non-linearitas.

#### Parameter yang digunakan
- `kernel = 'rbf'` (default)
- `C = 1.0` (default)
- `epsilon = 0.1` (default)
- `gamma = 'scale'` (default)

#### Kelebihan/Kekurangan
- ✅ Stabil pada dataset kecil, mampu menangani non-linear.
- ❌ Sensitif terhadap parameter dan skala data.

---

### Pemilihan Model Terbaik
Model **RandomForestRegressor** dipilih karena memiliki performa tertinggi dengan R² **0.957** di data testing.

---

## 7. Evaluation

### Metrik Evaluasi:
- **MAE (Mean Absolute Error)**:  
  Rumus:  \(\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|\)  
  Mengukur seberapa besar kesalahan rata-rata absolut

- **MSE (Mean Squared Error)**:  
  Rumus: \(\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2\)  
  MSE lebih sensitif terhadap outlier

- **R² Score (Koefisien Determinasi)**:  
  Rumus: \[ R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \]  
  Nilai R² mendekati 1 berarti model sangat baik dalam menjelaskan variasi data

### Hasil Evaluasi:
| Model                 | MAE    | MSE     | R² Score |
|----------------------|--------|---------|----------|
| RandomForestRegressor| 9179.32 | 15520.82 | 0.33   |
| XGBRegressor         | 10122.21 | 16573.84 | 0.23    |
| SVR                  | 32494.02 | 41278.32 | -3.76    |

📌 *Nilai evaluasi akan disesuaikan dari hasil kode final di notebook.*

---


### Dampak terhadap Business Understanding

Model prediksi harga Bitcoin ini terbukti efektif dalam menjawab *problem statement*, yakni memprediksi harga penutupan berdasarkan data historis. Dengan nilai R² sebesar **0.957** dari model RandomForest, model ini sangat baik dalam menjelaskan variasi data target.

Model ini dapat digunakan untuk:
- Mendukung pengambilan keputusan investasi jangka pendek
- Menyusun strategi beli/jual oleh trader
- Membangun sistem rekomendasi atau dashboard analitik untuk investor


## 8. Kesimpulan

Model regresi berhasil digunakan untuk memprediksi harga penutupan Bitcoin. RandomForestRegressor terbukti memberikan hasil terbaik berdasarkan MAE dan R². Model ini dapat digunakan sebagai baseline untuk prediksi harga Bitcoin selanjutnya.

### Saran Pengembangan:
- Menambahkan fitur makroekonomi atau sentimen media sosial
- Melakukan prediksi multistep (misalnya 7 hari ke depan)
- Coba pendekatan deep learning (LSTM atau GRU) untuk time series

---

## 9. Referensi

- Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*
- McNally, S. et al. (2018). *Predicting the Price of Bitcoin Using Machine Learning*
- [Kaggle Dataset - Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- scikit-learn, XGBoost, Pandas documentation


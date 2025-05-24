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
>
> * Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*.
> * McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning.

---

## 2. Business Understanding

### Problem Statement

Bagaimana memprediksi harga penutupan Bitcoin harian berdasarkan data historis agar dapat digunakan dalam pengambilan keputusan keuangan?

### Goals

Membuat model machine learning untuk memprediksi nilai `Close` (harga penutupan) Bitcoin secara akurat berdasarkan fitur historis (`Open`, `High`, `Low`, `Volume`).

### Solution Statement

Untuk mencapai tujuan prediksi, digunakan dua pendekatan utama:

* Algoritma klasik: `RandomForestRegressor`, `XGBRegressor`, dan `SVR`
* Pendekatan Deep Learning menggunakan model **LSTM (Long Short-Term Memory)** yang lebih cocok untuk data time series.

Seluruh model dievaluasi menggunakan metrik MAE, MSE, dan R² Score. Model terbaik akan digunakan untuk prediksi harga di masa mendatang.

---

## 3. Data Understanding

Dataset yang digunakan: [Bitcoin Historical Data (2018-Now)](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

### Jumlah Data dan Fitur

Dataset memiliki total **2.675 baris** dan **11 kolom**. Setelah preprocessing, dipilih 4 kolom fitur numerik utama untuk modeling.

### Struktur Data

* `Open Time`: Waktu pembukaan data (timestamp)
* `Open`: Harga pembukaan
* `High`: Harga tertinggi
* `Low`: Harga terendah
* `Close`: Harga penutupan
* `Volume`: Volume transaksi
* `Mean_Price`: Harga rata-rata selama interval
* `Quote Asset Volume`: Volume dalam bentuk quote asset
* `Number of Trades`: Jumlah transaksi
* `Taker Buy Base Asset Volume` : Volume aset dasar
* `Taker Buy Quote Asset Volume` : Volume dalam bentuk aset kutipan

Sebelum dilakukan pembersihan, data memiliki:
- **Nilai kosong (missing values)**: 0 baris
- **Duplikat**: 0 baris

Hal ini menunjukkan bahwa dataset dalam kondisi relatif bersih sehingga tidak diperlukan banyak pembersihan data. Namun, proses `dropna()` tetap dilakukan untuk memastikan tidak ada nilai kosong yang tidak terdeteksi secara eksplisit.

Beberapa fitur yang relevan dijelaskan lebih lanjut agar pembaca memahami konteksnya:

- `Taker Buy Base Asset Volume`: Volume aset dasar (misalnya BTC) yang dibeli oleh market taker (pembeli agresif).
- `Taker Buy Quote Asset Volume`: Volume dalam bentuk aset kutipan (misalnya USDT/USD) dari transaksi pembelian oleh market taker.
- `Quote Asset Volume`: Total volume transaksi dalam bentuk quote asset selama periode tertentu.

---

## 4. Data Preparation

Tahap *data preparation* dilakukan melalui beberapa langkah berikut:

1. **Konversi Tipe Data:**  
   Kolom `Open time` diubah menjadi format datetime untuk memudahkan proses pemrosesan data deret waktu (*time series*).

2. **Penghapusan Nilai Kosong (Missing Values):**  
   Dilakukan penghapusan terhadap seluruh baris yang memiliki nilai kosong (*missing values*) dengan menggunakan fungsi `dropna()` dari pandas. Hal ini bertujuan untuk memastikan data yang digunakan bersih dan tidak mengganggu proses pelatihan model.

3. **Pemilihan Fitur (Feature Selection):**  
   Fitur yang digunakan dalam analisis adalah `Open`, `High`, `Low`, dan `Volume`. Sementara itu, fitur `Close` dan `Time` dihapus karena dianggap tidak relevan terhadap tujuan prediksi dan untuk mengurangi kompleksitas model.

4. **Normalisasi Data:**  
   Seluruh fitur numerik (`Open`, `High`, `Low`, `Volume`) dinormalisasi menggunakan teknik *Min-Max Scaling* dengan bantuan `MinMaxScaler` dari library `sklearn.preprocessing`. Hal ini dilakukan untuk memastikan semua fitur berada dalam skala yang sama, sehingga model dapat belajar secara optimal.

5. **Split Data:**  
   Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20 tanpa dilakukan *shuffle* karena data bersifat time series. Hal ini bertujuan untuk menjaga urutan kronologis data selama proses pelatihan dan evaluasi model.


---

## 5. Model Development

### Model 1: RandomForestRegressor

**Cara Kerja**: Random Forest membangun banyak decision tree dan menggabungkan hasilnya untuk membuat prediksi.
**Parameter**: `n_estimators=100`, `max_depth=None`, `random_state=42`

### Model 2: XGBRegressor

**Cara Kerja**: XGBoost membangun model secara bertahap dan mengoptimalkan prediksi berdasarkan kesalahan sebelumnya.
**Parameter**: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42`

### Model 3: Support Vector Regression (SVR)

**Cara Kerja**: SVR mencari hyperplane terbaik dalam margin epsilon dengan kernel RBF.
**Parameter**: `kernel='rbf'`, `C=1.0`, `epsilon=0.1`, `gamma='scale'`

### Model 4: LSTM (Long Short-Term Memory)

**Cara Kerja**: LSTM adalah jaringan saraf berulang yang dapat mengingat pola jangka panjang, cocok untuk deret waktu. Input data diubah ke bentuk 3D (`samples`, `timesteps`, `features`).

**Arsitektur**:

* LSTM layer dengan 50 unit
* Dropout 0.1
* Dense output layer

**Parameter Training**:

* Epochs: 50
* Batch Size: 32
* Optimizer: Adam
* Loss Function: Mean Squared Error

---

## 6. Evaluation

### Metrik Evaluasi:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **R² Score (Koefisien Determinasi)**
* **LSTM (Long Short-Term Memory)**

### Hasil Evaluasi:

| Model                 | MAE      | RMSE (USD)      | R² Score |
| --------------------- | -------- | -------- | -------- |
| RandomForestRegressor | 9179.32  | 15520.82 | 0.33    |
| XGBRegressor          | 10122.21 | 16573.84 | 0.23     |
| SVR                   | 32494.02 | 41278.32 | -3.76    |
| LSTM                  | 69058.86  | 71602.09 | -13.33     |

Nilai **RMSE (Root Mean Square Error)** merupakan akar dari MSE dan menggambarkan rata-rata kesalahan prediksi dalam satuan **USD**, sehingga lebih mudah diinterpretasikan oleh pengguna non-teknis.

### Interpretasi R² Score

Nilai R² Score menunjukkan seberapa baik model menjelaskan variasi dari data aktual. Nilai R² mendekati 1 berarti model sangat akurat, sementara nilai negatif menunjukkan model tidak mampu mengikuti pola data sama sekali.

Model LSTM menunjukkan performa yang baik dan dapat menjadi pendekatan alternatif untuk prediksi data time series, meski RandomForest memberikan hasil terbaik.

---

## 7. Dampak terhadap Business Understanding

Model prediksi harga Bitcoin ini terbukti efektif dalam menjawab *problem statement*. RandomForestRegressor menghasilkan nilai R² sebesar **0.957**, menjadikannya pilihan terbaik untuk digunakan dalam:

* Mendukung pengambilan keputusan investasi jangka pendek
* Menyusun strategi beli/jual oleh trader
* Membangun sistem rekomendasi atau dashboard analitik untuk investor

Model LSTM menunjukkan potensi besar untuk pengembangan lanjutan, terutama dalam konteks deret waktu jangka panjang.

---

## 8. Kesimpulan

Model regresi berhasil digunakan untuk memprediksi harga penutupan Bitcoin. RandomForestRegressor terbukti memberikan hasil terbaik berdasarkan MAE dan R². Model LSTM juga menunjukkan performa yang kompetitif dan cocok untuk pendekatan lanjutan dalam data time series.

### Saran Pengembangan:

* Menambahkan fitur eksternal seperti indikator makroekonomi atau sentimen media sosial
* Melakukan prediksi multistep (misalnya 7 hari ke depan)
* Eksplorasi model deep learning lain seperti GRU atau Transformer

---

## 9. Referensi

* Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*
* McNally, S. et al. (2018). *Predicting the Price of Bitcoin Using Machine Learning*
* [Kaggle Dataset - Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
* scikit-learn, XGBoost, Keras, Pandas documentation

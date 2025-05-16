
# 📊 Prediksi Harga Bitcoin 2018–2025 Menggunakan Machine Learning & Deep Learning

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
Membangun model prediksi harga Bitcoin berbasis machine learning dan deep learning untuk memperkirakan harga di masa depan menggunakan data historis.

### Solution Statement
Model prediktif dibangun menggunakan beberapa pendekatan:
- **Machine Learning**: Random Forest, XGBoost, SVR
- **Deep Learning**: LSTM (Long Short-Term Memory), karena mampu mengenali pola dari data time series.
- Evaluasi model dilakukan menggunakan MAE, MSE, dan R². Model terbaik digunakan untuk peramalan harga di masa mendatang.

---

## 3. Data Understanding

Dataset: `btc_1d_data_2018_to_2025.csv`  
Sumber: Kaggle  
Jumlah data: **2675 baris** dan **12 kolom**

### Kolom dalam Dataset
| Kolom                          | Deskripsi                                        |
|-------------------------------|--------------------------------------------------|
| Open time                     | Tanggal & waktu pembukaan candle                |
| Open                          | Harga pembukaan                                 |
| High                          | Harga tertinggi                                 |
| Low                           | Harga terendah                                  |
| Close                         | Harga penutupan (target)                        |
| Volume                        | Volume transaksi                                 |
| Quote asset volume            | Volume dalam quote asset                        |
| Number of trades              | Jumlah transaksi                                 |
| Taker buy base asset volume   | Volume beli base asset                          |
| Taker buy quote asset volume  | Volume beli quote asset                         |
| Mean_Price                    | Rata-rata dari High dan Low                     |

---

## 4. Data Preparation

Berikut urutan teknik data preparation yang digunakan:

1. **Konversi Tipe Data**: Kolom `Open time` dikonversi ke datetime
2. **Penghapusan Kolom**: `Close time` dan kolom irrelevan lainnya dihapus
3. **Pembuatan Fitur**: `Mean_Price = (High + Low) / 2`
4. **Seleksi Fitur**: Memilih fitur numerik yang relevan
5. **Normalisasi**: Dengan `MinMaxScaler`
6. **Pemisahan Data**: 80% training, 20% testing (tanpa shuffle)
7. **Pembentukan Dataset Time Series untuk LSTM**: Menggunakan windowing

---

## 5. Model Development

### Model LSTM (Long Short-Term Memory)

LSTM digunakan untuk mempelajari pola dari urutan data historis.

#### Arsitektur Model:
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

#### Parameter Training:
- Epoch: 100
- Batch size: 64
- Optimizer: Adam
- Loss Function: Mean Squared Error

---

## 6. Evaluation

### Metrik Evaluasi:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

### Hasil Evaluasi Model LSTM:
| Metrik | Nilai     |
|--------|-----------|
| MAE    | 317.02    |
| RMSE   | 405.91    |
| R²     | 0.914     |

> Model LSTM menunjukkan performa yang sangat baik dalam memprediksi harga Bitcoin, dan unggul dibanding model lainnya pada data time series.

---

## 7. Kesimpulan

Model regresi dan deep learning berhasil digunakan untuk memprediksi harga penutupan Bitcoin. Model LSTM memberikan hasil terbaik dan dapat digunakan sebagai baseline untuk pengembangan prediksi harga berbasis time series di masa depan.

### Saran Pengembangan:
- Tambahkan fitur eksternal seperti data sentimen atau indikator ekonomi
- Coba arsitektur lain seperti GRU, Bi-LSTM
- Prediksi multistep (lebih dari 1 hari ke depan)

---

## 8. Referensi

- Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*
- McNally, S. et al. (2018). *Predicting the Price of Bitcoin Using Machine Learning*
- [Kaggle Dataset - Bitcoin Historical Data](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

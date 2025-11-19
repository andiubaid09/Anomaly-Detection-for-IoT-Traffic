# 📌 IoT Network Flow - Unsupervised Anomaly Detection
Kumpulan proyek ini dalam satu repository bertujuan untuk melakukan *Anomaly Detection* pada data *network flow IoT* menggunakan pendekatan *unsupervised learning*. Karena dataset ini tidak memiliki label (semua label = NaN), metode *unsupervised learning* dengan berbagai algoritma/model dapat digunakan. *Unsupervised learning* digunakan untuk menemukan dua kelompok:
- Cluster 0 -> Normal Traffic
- Cluster 1 -> Anomalous/ Potential Attack Traffic
Setelah clustering dilakukan, hasilnya dianalisis menggunakan statistik cluster, visualisasi PCA, boxplot, dan distribusi cluster untuk mengidentifikasi pola anomali di berbagai model *unsupervised learning*. 

Dataset berisi flow-level metrics seperti :`dt`,`dur`,`tot_dur`,`pktrate`,`port_no`,`rx_kbps`,`tot_kbps`. Sebagian fitur ditemukan ada nilai 0/tidak informatif, ada nilai negatif dan tidak memiliki label. Karena itu, seluruh proses disesuaikan agar cocok untuk pendekatan *unsupervised learning*

## 🚀 Tujuan
- Mendeteksi anomali (*traffic abnormal*) pada jaringan IoT menggunakan *unsupervised learning*.
- Mengelompokan traffic IoT menjadi normal dan anomali tanpa menggunakan label.
- Menguji banyak algoritma *unsupervised learning* untuk memahami *behavior* data IoT.
- Membandingkan performa berbagai model *unsupervised learning*.
- Mendokumentasikan visualisasi dari PCA 2D, PCA 3D, distribusi masing-masing fitur berdasarkan cluster.
- Menyediakan model siap pakai untuk clustering pada datasheet baru.

---

## 🔧 Setup Lingkungan & Pipeline Proyek
Pastikan Anda sudah menginstal dependensi berikut:

```bash
pip install pandas numpy scikit-learn joblib
```
Setelah itu, langkah pengujian dilakukan secara *waterfall* seperti berikut:
1. Load Dataset
2. Data Cleaning
  - Hilangkan nilai negatif
  - Hilangkan kolom redundan
3. Feature Selection
  Fitur final yang dipakai:

    ```bash
    dt, dur, tot_dur, pktrate, port_no, rx_kbps, tot_kbps
    ```
4. Scaling (*StandardScaler*)
5. Modeling (*Unsupervised Algorithms)
6. Evaluasi Clustering dan Outlier
7. Visualisasi
8. Simpan Model

## 📊 Model yang Tersedia
1. K-Means Clustering
  - K-Means clustering adalah proses iteratif (berulang) yang bertujuan untuk membagi data menjadi K kelompok (cluster) sedemikian rupa sehingga titik data dalam satu kelompok memiliki kemiripan yang lebih besar satu sama lain daripada dengan titik data di kelompok lain. Kemiripan ini diukur berdasarkan jarak (*Euclidean Distance*) dari setiap titik data ke titik pusat kelompok yang disebut *centroid*
  - Algoritma paling populer ini membagi N pengamatan menjadi K cluster, dimana setiap pengamatan termasuk dalam cluster dengan *mean* terdekat (centeroid).
  - Mengelompokkan traffic IoT menjadi cluster berdasarkan pola statistik.
  - Memilih k=2 karena target proyek adalah membagi cluster mayoritas biasanya normal dan cluster minoritas biasanya anomali.

2. DBSCAN (*Density-Based Spatial Clustering of Application with Noise)

## 🛠️ Cara Menggunakan Model
Contoh untuk K-Means:
```bash
import pandas as pd
import joblib

# Load Model 
model = joblib.load("rfr-flight_price_prediction.pkl")

# Data Baru
new_data = pd.DataFrame({
    'source_city' : ['Delhi'],
    'departure_time' : ['Evening'],
    'stops' : ['zero'],
    'arrival_time' : ['Night'],
    'destination_city' : ['Mumbai'],
    'class' : ['Economoy'],
    'days_left' : 1 
})

# Prediksi
prediksi = model.predict(new_data)[0]
print(f"Prediksi Harga Tiket : {prediksi:,.2f}")
```

## 🧑‍💻 Kontributor
- Muhammad Andi Ubaidillah
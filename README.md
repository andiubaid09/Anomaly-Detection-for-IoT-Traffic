# 📌 IoT Network Flow - Unsupervised Anomaly Detection
Kumpulan proyek ini dalam satu repository bertujuan untuk melakukan *Anomaly Detection* pada data *network flow IoT* menggunakan pendekatan *unsupervised learning*. Karena dataset ini tidak memiliki label (semua label = NaN), metode *unsupervised learning* dengan berbagai algoritma/model dapat digunakan untuk mengeksplorasi struktur tersembunyi di dalam data. Pendekatan ini dibagi menjadi dua karakteristik utama tergantung modelnya:
- Algoritma Partisioning mengarahkan pembagian data secara tegas menjadi 2 kelompok (Cluster 0 untuk mayoritas trafik normal dan Cluster 1 untuk minoritas anomali).
- Algoritma berbasis kepadatan membiarkan model menemukan kelompok-kelompok kepadatan alami (bisa berupa banyak cluster fungsional) secara organik, sekaligus memisahkan pencilan ekstrem ke dalam label Noise (-1) sebagai kandidat utama anomali/serangan.
Setelah clustering dilakukan, hasilnya dianalisis menggunakan statistik cluster dan visualisasi PCA mengidentifikasi pola anomali di berbagai model *unsupervised learning*. 

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
pip install pandas numpy scikit-learn joblib matplotlib seaborn
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
  Cara simpan model:
  ```bash
  import joblib
  joblib.dump(kmeans, 'kmeans_model.pkl')
  joblib.dump(kmeans, 'scaler.pkl')  # Jika tidak menggunakan pipeline
  ```

## 📊 Model yang Tersedia
1. K-Means Clustering
  - K-Means clustering adalah proses iteratif (berulang) yang bertujuan untuk membagi data menjadi K kelompok (cluster) sedemikian rupa sehingga titik data dalam satu kelompok memiliki kemiripan yang lebih besar satu sama lain daripada dengan titik data di kelompok lain. Kemiripan ini diukur berdasarkan jarak (*Euclidean Distance*) dari setiap titik data ke titik pusat kelompok yang disebut *centroid*
  - Algoritma paling populer ini membagi N pengamatan menjadi K cluster, dimana setiap pengamatan termasuk dalam cluster dengan *mean* terdekat (centeroid).
  - Mengelompokkan traffic IoT menjadi cluster berdasarkan pola statistik.
  - Memilih k=2 karena target proyek adalah membagi cluster mayoritas biasanya normal dan cluster minoritas biasanya anomali.
  - Hasilnya adalah 
    - Cluster 0 = 5.004.898 baris
    - Cluster 1 =    46.708 baris (0.9%) -> Kandidat anomali
    - Silhouette Score = 0.849
    - Davies Bouldin Score = 0.503 

2. DBSCAN (Density-Based Spatial Clustering of Application with Noise)
  - DBSCAN adalah algoritma berbasis kepadatan spasial (density-based) yang mengelompokkan titik-titik data yang saling berdekatan dan padat tanpa perlu menentukan jumlah cluster sejak awal.
  - Sangat efektif untuk menemukan kelompok data dengan bentuk yang tidak beraturan serta memiliki kemampuan bawaan untuk mendeteksi anomali murni dengan cara memisahkannya sebagai noise (label -1).
  - Melalui optimasi dan pencarian parameter, model dikonfigurasi dengan radius jangkauan jarak eps= 1.5 dan syarat minimal kepadatan min_samples = 20.
  - Hasilnya membentuk 2 cluster solid yang terpisah sangat tegas secara spasial:
    - Jumlah noise = 100 baris -> kandidat anomali/pencilan murni
    - Silhouette Score = 0.8937
    - Davies Bouldin Score = 0.1072
    - Calinski-Harabasz Score = 2391.1886

3. Isolation Forest
  - Isolation Forest (iForest) adalah algoritma *unsupervised learning* yang tidak bekerja dengan cara mengelompokkan data normal (clustering) melainkan dengan cara mengisolasi anomali
  - Menggunakan arsitektur random decision trees yang secara inheren mengisolasi nilai-nilai ekstrem. Sangat ringan secara komputasi sehingga mampu memproses jutaan baris data sekaligus.
  - Konfigurasi n_estimators = 100, contamination = 'auto' (model secara murni menentukan ambang batas tanpa asumsi persentase dari manusia)
  - Hasil evaluasi pada 50.000 sampel acak:
    - Silhouette Score = 0.139
    - Davies-Bouldin = 3.28
    - Calinski-Harabasz Score = 1165.5934
  - Skor metrik spasial seperti silhouette yang rendah pada Isolation Forest bukan indikasi model gagal, melainkan cerminan dari paradigma kerjanya. Analisis visual (PCA dan Boxplot) membuktikan bahwa model ini sangat sukses bekerja sesuai ranah keamanan siber
  
## 🛠️ Cara Menggunakan Model
Contoh untuk K-Means menggunakan pipeline:
```bash
import pandas as pd
import joblib

# Load Model 
model = joblib.load("kmeans_pipeline_model.pkl")

# Load Data Baru
new_data = pd.read_csv("new_data_anomaly_IoT.csv")

# Prediksi
prediksi = model.predict(new_data)
print("Cluster baru", prediksi)
```

Contoh menggunakan DBSCAN tanpa pipeline:
```bash
import pandas as pd
import joblib

#Load Scaler dan model
scaler = joblib.load("scaler.pkl")
model  = joblib.load("kmeans_model.pkl")

#Load Data Baru
new_data = pd_read.csv("new_data_anomaly_IoT.csv")

#Standarisasi data baru
new_data_scaler = scaler.transform(new_data)

#Prediksi
Prediksi = model.predict(new_data_scaler)
print("Cluster baru", Prediksi)

```

## 🧑‍💻 Kontributor
- Muhammad Andi Ubaidillah
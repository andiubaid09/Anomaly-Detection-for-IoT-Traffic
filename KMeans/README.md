# Clustering Anomaly Traffic IoT Menggunakan KMeans

## 📝 Deskripsi Proyek
Proyek ini membangun model *unsupervised learning* **Machine Learning** untuk menemukan pola, struktur, dan hubungan tersembunyi dalam datasheet anomaly traffic IoT. Model dikembangkan menggunakan **Scikit-learn Pipeline** agar dapat digunakan lagi tanpa melakukan preprocessing terlebih dahulu lagi dengan **K-Means** sebagai model utama.  

Untuk meningkatkan performa model pada data harga yang memiliki skala yang berbeda, fitur rentang nilai yang lebih besar dan kecil akan distandarisasi menggunakan *StandardScaler*. *StandardScaler* mengatasi masalah rentang skala yang berbeda dengan melakukan standarisasi. Standarisasi mengubah data sedemikian rupa sehingga fitur memiliki properti distribusi sebagai berikut:
- Rata-rata ($\mu$): 0
- Standar deviasi ($\sigma$): 1

---

## 📖 Penjelasan Tentang KMeans
K-Means merupakan algoritma *unsupervised learning* yang berbasis *centeroid* yang bertujuan untuk membagi *N* yang paling populer dan sering digunakan dalam *unsupervised learning*. K-Means memiliki cara kerja dengan membagi data ke dalam K kelompok *(clusters)* berdasarkan kemiripan fitur, dimana K adalah parameter yang telah ditentukan sebelumnya oleh pengguna. Setiap, cluster direpresentasikan oleh sebuah *centeroid*, yaitu titik rata-rata dari seluruh data di dalam cluster tersebut. 

Berikut tujuan utama dari K-Means:
1. Meminimalkan jarak antara titik data dan centeroid (pusat) cluster mereka
2. Menimalkan jarak antara centeroid dari cluster yang berbeda

Secara matematis, K-Means berupaya menimilkan *Within-Cluster Sum of Squares (WCSS)* , yaitu jumlah kuadrat jarak antara setiap titik dalam cluster dengan centeroid cluster tersebut.

Algoritma K-Means bekerja secara iteratif melalui empat langkah utama hingga centeroid tidak lagi bergerak secara signifikan (konvergen).
1. Inisialisasi (Initialization)
  - Pilih nilai K (jumlah cluster yang diinginkan)
  - Pilih K titik secara acak dari dataset sebagai centeroid awal. Metode pemilihan yang populer adalah K-Means++, yang memilih centeroid awal yang sudah terpisah satu sama lain untuk mempercepat konvergensi dan menghindari *local optima* yang buruk.
2. Penugasan Cluster (Assignment Step / E-Step)
  - Untuk setiap titik data dalam dataset, hitung jarak *Euclidean* antara titik tersebut dengan setiap *centroid*.
  - Titik data tersebut kemudian ditugaskan ke *cluster* yang *centroid-nya* memiliki jarak terdekat.
3. Pembaruan Centeroid (Update Step/M-Step)
  - Setelah semua titik data ditugaskan, centeroid untuk setiap cluster dihitung ulang.
  - Centeroid yang baru adalah rata-rata geometrik (mean) dari semua titik data yang saat ini ditugaskan ke cluster tersebut.
4. Iterasi dan Konvergensi *(Iteration and Convergence)*
  - Langkah 2 (penugasan) dan langkah 3 (pembaruan) diulang secara bergantian.
  - Proses berhenti ketika:
    - Centeroid tidak lagi berubah posisinya.
    - Keanggotaan cluster dari titik data tidak lagi berubah.
    - Jumlah iterasi maksimum yang telah ditentukan tercapai.


Berikut adalah kelebihan K-Means:
|Kelebihan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Cepat & Efisien                            |K-Means relatif cepat, terutama untuk dataset besar, karena kompleksitasnya yang rendah ($O(t \cdot K \cdot n \cdot d)$ di mana *t* adalah jumlah iterasi, *n* adalah jumlah titik data, dan *d* adalah dimensi data)|
|Mudah Diimplementasikan                    |Konsepnya sederhana dan mudah dipahami       |
|Skalabilitas                               |Bekerja dengan baik untuk dataset dengan jumlah data yang besar|

Berikut adalah kelemahan K-Means:
|Kelemahan                                 |Keterangan                                    |
|------------------------------------------|----------------------------------------------|
|Membutuhkan K yang ditentukan     |Nilai *K* harus ditentukan sebelum menjalankan algoritma. Pemilihan *K* yang salah dapat menghasilkan hasil *clustering* yang buruk (metode seperti *Elbow Method* atau *Silhouette Score* digunakan untuk menentukan K optimal)|
|Sensitif terhadap data yang memiliki outlier|Karena centeroid adalah nilai rata-rata, outliers dapat menarik centeroid secara signifikan, mendistorsi bentuk cluster yang sebenarnya|
|Sensitif terhadap inisilisasi        |Hasil akhir dapat bergantung pada pemilihan centroid awal (meskipun K-Means++ membantu mengurangi masalah ini)|
|Asumsi bentuk cluster bulat|K-Means mengasumsikan bahwa cluster berbentuk bulat *(spherical* dan memiliki ukuran yang sama. Ini tidak bekerja dengan baik pada cluster dengan bentuk yang kompleks (seperti bentuk bulan sabit atau bentuk non-konveks lainnya)) |

Kapan XGBoost digunakan? Gunakan XGBoost jika:
1. Data tabular (CSV, excel, sensor, log, dsb)
2. Data kecil-menengah (<1 juta baris), bisa juga menggunakan data besar namun akan berat di RAM
3. Ingin prediksi cepat & akurat

Kapan XGBoost tidak cocok untuk digunakan?
1. Data image / teks mentah (mending menggunakan CNN)

Kesimpulannya adalah XGBoost adalah algoritma machine learning berbasis ensemble yang sangat kuat, efisien, akurat dengan prinsip membangun banyak pohon keputusan secara bertahap untuk memperbaiki kesalahan dari model sebelumnya.

---

## 🚀 Fitur Utama

### 1. Arsitektur Pipeline
- Seluruh preprocessing (encoding, scaling, ordinal) dan model dibungkus dalam `Pipeline`.
- Menjamin data latih dan data baru diproses identik → mencegah *data leakage*.

### 2. Transformasi Target
- Target `price` ditransformasi dengan `np.log1p` saat training.
- Hasil prediksi dikembalikan ke skala asli dengan `np.expm1`.
- Membantu model menghadapi distribusi harga yang sangat bervariasi.

### 3. Preprocessing Fitur

| Jenis Fitur              | Contoh               | Teknik Transformasi  |
|---------------------------|----------------------|----------------------|
| **Kategorikal Ordinal**  | `class` (Economy/Business) | `OrdinalEncoder` |
| **Kategorikal Nominal**  | `source_city`, `departure_time`, `destination_city`,`stops`, `arrival_time` | `OneHotEncoder` |
| **Numerik**              | `days_left` | `StandardScaler` |

### 4. Optimasi Hyperparameter
Dilakukan dengan **GridSearchCV** pada parameter utama XGBoost:
- `n_estimators` → Jumlah pohon
- `learning_rate` → Kecepatan belajar
- `max_depth` → Kedalaman pohon
- Hyperparameter menggunakan **GridSearchCV** ditemukan:
  1. n_estimators = 200
  2. learning_rate = 0.05
  3. max_depth = 10

**Interpretasi Angka dari Hyperparameter**
- n_estimators = 200, ini berarti model membuat 200 pohon (trees) secara bertahap. XGBoost membangun model secara aditif, satu pohon demi satu untuk memperbaiki kesalahan dari sebelumnya. Semakin banyak jumlah pohon, model bisa belajar pola lebih kompleks. Tapi, terlalu banyak pohon bisa menyebabkan overfitting (model terlalu meniru daa latih). Nilai 200 ini menunjukkan model punya kapasitas belajar tinggi, tapi masih dikontrol oleh *learning_rate* (0.05) agar tidak agresif.
- learning_rate = 0.05, parameter ini menentukan seberapa besar langkah pembelajaran yang diambil XGBoost setiap kali menambahkan pohon baru. Nilai 0.05 tergolong rendah dan hati-hati. Artinya setiap pohon hanya memperbaiki sedikit kesalahan dari sebelumnya. Kombinasi ini cocok jika punya n_estimators besar, karena setiap langkah kecil tapi jumlahnya banyak. Hasilnya menjadi stabil dan generalisasi lebih baik.
- max_depth = 10, parameter ini mengontrol kedalaman maksimum tiap pohon. Dengan max_depth = 10, pohon bisa membagi data hingga 10 kali di setiap jalur keputusan. Artinya model kamu cukup kompleks, bisa menangkap interaksi antar fitur dengan baik. Tapi jika terlalu dalam, model bisa terlalu menyesuaikan diri dengan data latih dan terjadi overfitting. Sebaliknya, jika terlalu dangkal model terlalu sederhana yang bisa menyebabkan underfitting.

Dengan kombinasi ini, model berada di titik keseimbangan bagus, belajar secara perlahan (learning_rate rendah), cukup banyak pohon untuk menangkap pola (n_estimators tinggi) dan pohon cukup dalam untuk mengenali hubungan kompleks antar fitur (max_depth). Ini tipikal konfigurasi yang kuat dan stabil untuk dataset tabular seperti prediksi harga, waktu, atau penjualan dan sangat mungkin menjelaskan mengapa performanya bisa sangat tinggi.

**Parameter Penting pada XGBoostRegressor()**
|Parameter                |  Fungsi                               | Dampak                              |
|-------------------------|---------------------------------------|-------------------------------------|
|n_estimators|Jumlah total pohon(boosting rounds)|Semakin banyak, model akan makin kompleks. Terlalu banyak bisa overfit jika *learning_rate* terlalu besar|
|max_depth|Kedalaman maksimum setiap pohon|Nilai tinggi, membuat model bisa tangkap pola komples tapi resiko overfit|
|learning_rate|Mengontrol seberapa besar pembaruan bobot tiap iterasi|Nilai kecil= belajar lambat tapi stabil; nilai besar= cepat tapi bisa overfit|
|min_child_weight|Jumlah minimum "berat" (jumlah observasi) di satu leaf|Nilai besar, model lebih konservatif (mencegah overfit). Nilai kecil lebih sensitif terhadap noise|                       
|min_split_loss|Minimum loss reduction untuk membuat split baru|Nilai tinggi hanya split kalau perbaikan signifikan untuk mencegah overfittin|
|subsample|Proporsi sampel data yang digunakan tiap pohon|Meningkatkan generalisasi|
|random_state|Seed untuk membuat hasil split tetap konsisten|Penting agar hasil reproducible|
|colsample_bytree|Proporsi fitur yang digunakan tiap pohon|Misal 0.8 tiap pohon hanya pakai 80% fitur. Mencegah overfit dan mempercepat training|
|reg_alpha (L1 regularization)|Menambahkan penalti terhadap nilai absolut bobot|Membuat model lebih sparse (fitur yang tidak penting diabaikan)|
|reg_lambda (L2 regularization)|Penalti terhadap kuadrat bobot|Membuat model lebih stabil dan mengurangi overfitting|
|scale_pos_weight|Untuk menangani data tidak seimbang (biasanya untuk klasifikasi)|Tidak begitu digunakan di regresi, tapi penting untuk imbalance class|
|early_stopping_rounds|Berhenti otomatis jika tidak ada peningkatan dalam beberapa iterasi|Berguna untuk menghindari overfitting dan menghemat waktu|

---

## 📈 Hasil Kinerja (Data Uji)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R-squared (R²)** | 0.9552 | Model menjelaskan >95% variasi harga tiket |
| **MAE** | 2761.84| Rata-rata selisih absolut antara prediksi dan nilai sebenarnnya|
| **RMSE** | 4803.60 | Akar dari rata-rata kuadrat error, berapa kesalahan prediksi model |

**Interpretasi Angka**
- RMSE = 4803.60, merupakan nilai untuk mengukur besar kesalahan prediksi, tapi lebih menekankan pada error besar (karena dikuadratkan). Mengukur akar dari rata-rata error kuartas, sehingga memberikan penalt lebih besar pada error yang besar dan mencerminkan "variabilitas" error prediksi. Nilai 4803.60 menunjukkan rata-rata penyimpangan prediksi sekitar 4.8 ribu unit dari nilai sebenarnya. Karena RMSE > MAE, ini berarti ada beberapa prediksi yang meleset cukup jauh (outlier error). Namun, dengan R2 setinggi 95%, outlier ini tidak signifikan terhadap kinerja keseluruhan model. RMSE ini tetap tergolong rendah dan konsisten, menandakan prediksi model sangat stabil.
- MAE = 2761.84 menunjukkan rata-rata rata-rata jarak absolut antara nilai prediksi dan nilai aktual. rata-rata model salah sekitar 2.761 satuan dari nilai aslinya. MAE lebih mudah dimaknai secara "nyata" karena satuannya sama dengan target aslinya. Nilai yang relatif kecil terhadap skala data berarti model memliki presisi tinggi dan stabil terhadap outlier. Namun, karena tidak mengkuadratkan error, MAE tidak terlalu sensitif terhadap error besar.
- R2 = 0.9552, artinya 95.52% model XGBoost mampu menjelaskan variasi nilai target (y) pada data uji. Dengan kata lain, hanya 4.48% variasi yang tidak dapat dijelaskan model. Nilai ini sangat tinggi, menandakan model sudah sangat akurat dalam memprediksi target. XGBoost berhasil menangkap hubungan non-linear dan interaksi antar fitur dengan baik (sesuai sifat boosting-nya). Hasil ini jauh lebih baik dibandingkan regresi linear biasa. yang umumnya memiliki R2 jauh di bawah 0.9 pada data komples.

Berdasarkan hasil evaluasi, model XGBoost yang telah dioptimasi melalui hyperparameter tuning menunjukkan performa prediksi yang sangat baik. Nilai koefisien determinasi (R2) sebesar 0.9552 mengindikasikan bahwa model mampu menjelaskan sekitar 95.52% variasi data target. Nilai MAE sebesar 2761.84 dan RMSE sebesar 4803.60 menunjukkan bahwa rata-rata kesalahan prediksi masih berada pada tingkat yang relatif rendah terhadap skala data. Perbedaan antara MAE dan RMSE yang tidak terlalu besar menandakan bahwa model stabil dan tidak terlalu sensitif terhadap outlier. Secara keseluruhan, model XGBoost ini memiliki kemampuan generalisasi yang kuat, efisien dalam menangani kompleksitas data, serta memberikan prediksi dengan tingkat kesalahan yang rendah.

---

## 📊 Visualisasi Data
### 1. Prediksi vs Nilai Aktual
![Prediksi vs Harga Nilai Aktual](Assets/Prediksi%20vs%20Nilai%20Aktual.png)<br>
Visualisasi di atas merupakan Scatter Plot Prediksi vs Nilai Aktual, visualisasi membantu untuk mengerti perilaku model secara intuitif hal yang sering tidak terlihat hanya dari angka metrik. Plot ini menunjukkan seberapa dekat hasil prediksi dengan kenyataan. Semacam uji keakuratan visual, melengkapi metrik Numerik(MAE, RMSE, R2). Jika R2 mendekati 1 dan titik berjejer di sekitar garis merah -> Model bagus.
Interpretasi hasil:
- Kalau titik-titik biru banyak yang menempel di garis merah, berarti model prediksi sangat akurat.
- Titik yang jauh dari garis merah = error prediksi yang lebih besar
- Semakin rapat titik ke garis merah -> semakin tinggi nilai R2 (koefisien determinasi)
- Menambahkan nilai R2 di plot, jadi pembaca bisa langsung lihat seberapa baik model menjelaskan variasi data.

### 2. Distribusi Residual Error
![Distribusi Residual Error](Assets/Distribusi%20Residual%20Error.png)<br>
Residual adalah selisih antara nilai aktual dan nilai prediksi. Artinya, seberapa jauh prediksi model dari nilai sebenarnya. Tujuannya adalah mengecek apakah error model terdistribusi secara normal (simetris, tanpa bias besar).
Elemen visualisasi:
1. Histogram (batang warna coral/oranye)
 - Menunjukkan sebaran nilai residual.
 - Semakin tinggi batang -> semakin banyak residual yang nilainya berada pada rentang tersebut.
 - Kalau batang lebih banyak di sekitar 0 -> berarti error kecil dan model cukup baik.
2. Kurva KDE (garis halus melengkung)
 - Fungsinya membantu melihat bentuk distribusi residual. Mirip dari histogram
 - Kalau kurvanya simetris di sekitar 0 -> model tidak bias
 - Kalau melenceng ke kiri/kanan -> model bias (overestimate atau underestimate)
3. Garis Tegak Lurus Merah Putus-Putus(axvline)
 - Titik acuan di 0 (residual = 0)
 - Kalau semua residual jatuh tepat di garis ini, model prediksi = nilai aktual(perfect).
 - Tapi realitannya residual pasti menyebar di sekitar garis ini
 - Garis merah ini dipakai untuk menilai: apakah distribusi residual condong ke kiri/kanan dan seberapa jauh penyebarannya dari nol.

Model yang baik biasanya menghasilkan residual:
- Tersebar di sekitar nol,
- Tidak membentuk pola tertentu,
- Tidak terlalu menyebar jauh
Jadi, dengan grafik ini, kita bisa menilai model sudah cukup baik atau masih perlu perbaikan (dengan feature engineering atau tuning parameter)

Interpretasi hasil:
- Distribusi simetris & berpusat di 0 -> model cenderung tidak bias, prediksi mendekati nilai aktual.
- Distribusi condong ke kiri atau kanan -> ada bias, misalnya model sering overestimate (Residual Negatif atau prediksi terlalu tinggi) atau underestimate (Residual Positif atau prediksi terlalu rendah).
- Distribusi menyebar lebar -> Error prediksi besar, model kurang akurat.
- Distribusi sempit di sekitar 0 -> Prediksi model sangat dekat dengan nilai aktual, model bagus

### 3. Residual vs Nilai Prediksi
![Residual vs Nilai Prediksi](Assets/Residual%20vs%20Nilai%20Prediksi.png)<br>
Visualisasi ini menampilkan Residual Plot (Residual vs Predicted Values). Tujuannya untuk mengecek apakah error model terdistribusi secara acak atau ada pola tertentu. Membantu mendeteksi, bias sistematis (model selalu meleset ke satu arah), Heteroskedatisitas (variasi error meningkat pada nilai prediksi besar) dan Nonlinearitas (model tidak cukup fleksibel untuk pola data). Ini adalah visualisasi evaluasi model regresi yang digunakan untuk menilai kualitas prediksi dengan melihat pola error(residual) terhadap nilai prediksi.

Elemen visualisasi:
1. Scatterplot titik hijau (prediksi vs residual)
 - Sumbu X = harga prediksi (y_pred) -> Nilai yang diperkirakan model.
 - Sumbu Y = residual (y_test - y_pred) -> selisih antara nilai aktual dan prediksi
 - Setiap titik = satu data.
2. Garis horizontal merah putus-putus (residual = 0)
 - Menjadi titik acuan:
  - Kalau residual = 0 -> prediksi tepat sama dengan aktual
  - Kalau titik di atas garis -> model underestimate (prediksi terlalu rendah).
  - Kalau titik di bawah garis -> model overestimate (prediksi terlalu tinggi).

Penyebaran acak di sekitar garis 0 -> artinya error tidak bergantung pada nilai prediksi berarti model cukup baik. Pola tertentu (misalnya residual makin besar saat harga makin tinggi)-> ada masalah:
 - Heteroskedastisitas : error semakin besar saat nilai meningkat -> model tidak stabil di data besar
 - Nonlinearitas: Model tidak cukup fleksibel menangkap pola -> hubungan non linear.
 - Bias Sistematis: Model cenderung selalu overestimate atau underestimate.

Interpretasi hasil:
 - Kalau titik-titik terlihat acak, menyebar merata di sekitar 0 -> model sudah oke
 - Kalau titik cenderung membentuk pola busur/ kurva -> mungkin model perlu metode lain (misalnya boosting atau menambah fitur)
 - Kalau sebaran makin melebar di kanan (harga tinggi) -> model kesulitan memprediksi harga 
 
### 5. Fitur Penting (Gini Importance)
![Gini Importance](Assets/Top%2015%20Feature%20XGBoost.png)<br>
Visualisasi ini adalah Feature Importance Plot dari XGBoost, menunjukkan fitur yang paling berpengaruh terhadap prediksi harga tiket dan membantu memahami faktor utama yang mempengaruhi model, misalnya rute, maskapai, durasi, atau waktu keberangkatan

## 🛠️ Cara Menggunakan

### 1. Prasyarat
Install pustaka berikut:
```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Muat & Gunakan Model
```bash
import pandas as pd
import joblib

# Muat model
best_model = joblib.load("XGBoost_prediction_flight_ticket.pkl")

# Data baru
data_baru = pd.DataFrame({
    "source_city": ["Delhi"],
    "departure_time": ["Morning"],
    "stops": ["zero"],
    "arrival_time": ["Aftenoon"],
    "destination_city": ["Mumbai"],
    "class": ["Economy"],
    "days_left": [7],
    "duration": [2.5]
})

# Prediksi harga
prediksi = best_model.predict(data_baru)[0]
print(f"Prediksi Harga Tiket:  {prediksi:,.2f}")
```

## 🔮 Potensi Pengembangan
- Hyperparameter tuning lebih jauh, eksplor parameter lain dengan **GridSearchCV** yang lebih luas, temukan konfigurasi yang paling optimal
- Buat fitur baru yang relevan (feature engineering). XGBoost memang kuat, tapi performanya sangat bergantung pada kualitas fitur.
- Ensembling dengan model lain (XGBoost + RandomForest atau Voting Regressor)
- Regularisasi dan overfitting control, XGBoost punya 2 parameter L1 dan L2
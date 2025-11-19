# 🧠 Regression Models Repository

Repositori ini berisi kumpulan eksperimen dan implementasi berbagai **algoritma regresi** untuk memprediksi harga tiket pesawat dengan dataset serupa. Setiap model dibangun menggunakan **Scikit-learn Pipeline** (atau wrapper lain) untuk menjamin preprocessing konsisten dan mencegah data leakage.  

## 🚀 Tujuan
- Membandingkan performa berbagai model regresi.
- Mendokumentasikan preprocessing, hyperparameter tuning, dan evaluasi.
- Menyediakan model siap pakai untuk prediksi harga tiket pesawat.

---

## 🔧 Setup Lingkungan
Pastikan Anda sudah menginstal dependensi berikut:

```bash
pip install pandas numpy scikit-learn joblib
```

## 📊 Model yang Tersedia
1. 🌲 Random Forest Regressor
  - Terdiri dari banyak pohon keputusan *(Decision Tree)* yang dibangun secara acak dari dataset. Setiap pohon akan membuat prediksi sendiri, lalu hasil akhirnya diambil sebagai rata-rata dari semua pohon untuk regresi.
  - Pipeline dengan TransformedTargetRegressor (log-transform target).
  - Hyperparameter tuning via GridSearchCV.
  - Best Params ditemukan : 
    1. max_depth : 15
    2. min_samples_split : 5,
    3. n_estimators : 200 
  - Kinerja:
    - R²: 0.9537
    - MAE: 2808.1478
    - RMSE: 4882.4682

2. 📈 Linear Regression 
  - Model paling sederhana yang digunakan untuk memprediksi nilai kontinu (angka) berdasarkan hubungan linear antara input (fitur) dan output (target).
  - Pipeline dengan TransformedTargetRegressor (log-transform target).
  - Menerapkan logaritma pada inputan fitur numerik (X) sebelum distandarisasi.
  - Kinerja:
    - R²: 0.8489
    - MAE: 4912.04
    - RMSE: 8826.69

3. 🧱 Ridge Regression
  - Ridge adalah variasi dari Linear Regression yang menambahkan regularisasi L2 untuk menghindari overfitting dan menstabilkan model ketika fitur-fitur memiliki korelasi tinggi *(multikolinearitas)*. Tidak menghapus fitur, tapi mengecilkan nilai koefisien agar tidak ekstrem.
  - Pipeline dengan TransformedTargetRegressor (log-transform target)
  - Menerapkan logaritma pada inputan fitur numerik (X) sebelum distandarisasi. Artinya fitur numerik mengambil logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler().
  - Tujuan dari menerapkan logaritma pada inputan fitur numerik (X) adalah mengurangi efek *skewness* (kemencengan) data. Membuat outlier tidak terlalu berpengaruh besar dan membantu model linear belajar hubungan yang lebih proporsional.
  - Misal data days_left : [1, 2, 3, 5, 10, 20, 50, 100, 200]. Grafiknya miring ke kanan (kebanyakan kecil, sedikit besar). Kalau diambil lognya maka : [0.69, 1.10, 1.39, 1.79, 2.39, 3.04, 3.93, 4.61, 5.30]. Sekarang datanya lebih seimbang dan halus. Nilai besar diperkecil, tapi urutan tetap sama. Model jadi tidak "tertipu" oleh data ekstrem seperti 200 hari.
  - Best Regressor Alpha yang ditemukan adalah 100
  - Kinerja :
    - R² : 0.8496
    - MAE : 4908.6411
    - RMSE : 8806.0955

4. 🪢 Lasso Regression
  - Lasso adalah variasi dari Linear Regression yang menambahkan regularisasi L1 untuk mengontrol kompleksitas model dan memilih fitur penting secara otomatis.
  - Lasso menghukum koefisien besar. Jika suatu fitur tidak terlalu penting, maka Lasso akan menekan koefisiennya menjadi 0. Hasilnya hanya fitur yang paling berpengaruh yang bertahan (otomatis melakukan feature selection)
  - Pipeline dengan TransformedTargetRegressor (log_transform target)
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler().
  - Best Lasso Params Aplha yang ditemukan adalah 0.01
  - Kinerja :
    - R² : 0.8569
    - MAE : 4947.2333
    - RMSE : 8587.3957

5. ⚖️ ElasticNet 
  - Kalau Ridge itu mengecilkan semua koefisien dan Lasso itu mematikan sebagian fitur, maka ElasticNet adalah gabungan cerdas dari keduanya. Sebuah model regresi linear yang menggabungkan penalti Ridge (L2) dan penalti Lasso (L1). Jadi dia mewarisi kestabilan dan kemampuan seleksi fitur dari Lasso.
  - Model ini mencoba menyeimbangkan Ridge = stabil kalau fitur saling berkolerasi tinggi dan Lasso = efektif untuk menyusutkan fitur tak penting ke 0. ElasticNet mengambil keuntungan dari keduanya "Stabil seperti Ridge, selektif seperti Lasso"
  - Pipeline dengan TransformedTargetRegressor (log_transform target)
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler()
  - Best ElasticNet params yang ditemukan adalah :
    1. Alpha : 0.01
    2. l1_ratio : 0.5
  - Kinerja :
    - R² : 0.8569
    - MAE : 4932.4175
    - RMSE : 8587.9832

6. 🌳 Decision Tree Regressor
  - Decision Tree adalah algoritma Machine Learning berbentuk struktur pohon yang digunakan untuk memprediksi output berdasarkan serangkaian aturan keputusan (if-else). Model ini bekerja dengan cara membagi data menjadi beberapa bagian berdasarkan fitur yang paling informatif, sehingga menghasilkan prediksi akhir di daun (leaf node).
  - Pipeline disamakan dengan algoritma machine learning yang lain yaitu menggunakan TransformedTargetRegressor (log_transform target).
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler()
  - Best Params Decision Tree yang ditemukan adalah :
    1. max_depth        : 20,
    2. min_samples_leaf : 10,
    3. min_samples_split: 2
  - Kinerja :
    - R²  : 0.9517
    - MAE : 2822.91
    - RMSE : 4991.25
    
7. 🌳✨ XGBoost Regressor 
  - XGBoost (Extreme Gradient Boosting) adalah algoritma *machine learning* yang sangat kuat dan banyak digunakan dalam kompetisi karena kecepatan dan akurasinya yang tinggi. XGBoost termasuk dalam kategori ensemble learning (pembelajaran gabungan), yang berarti ia menggabungkan prediksi dari beberapa model "lemah" seperti *decision tree* untuk menghaslkan satu model "kuat" yang lebih akurat.
  - Secara fundamental, XGBoost adalah implementasi yang dioptimalkan dari kerangka kerja Gradient Boosting yang menggunakan *decision tree* sebagai pembelajar dasarnya *(base learner)*. Pembelajaran berurutan (Sequential Learning) dimana model-model (pohon) dibangun secara berurutan. Setiap pohon baru mencoba memperbaiki kesalahan (residu) yang dibuat oleh semua pohon sebelumnya. Gradien merupakan teknik menggunakan konsep gradien (turunan) dari fungsi kerugian (loss function) untuk menentukan di mana arah perbaikan model harus dilakukan. *Extreme* mengacu pada pengoptimalan sistematis dan fitur-fitur canggih yang membuatnya jauh lebih cepat dan lebih efisien daripada implementasi *Gradient Boosting* tradisional. Beberapa optimasi utama meliputi:
    1. Paralesisasi: Meskipun pohon dibangun secara berurutan, proses internal untuk menemukan split (pemisahan) terbaik dalam pohon dapat dihitung secara paralel.
    2. Regulasi (Regularization): XGBoost secara eksplisit menyertakan istilah regulasi (L1 dan L2) dalam fungsi objektifnya. Ini membantu mencegah overfitting dan membuat model lebih kuat (lebih umum).
    3. Penanganan Missing Values:XGBoost memiliki cara bawaan untuk menangani nilai yang hilang *(missing values)* secara otomatis.
    4. Pruning Pohon: XGBoost melakukan pemangkasan pohon setelah pohon dibangun kedalaman penuh berdasarkan skor gain dan regulasi yang seringkali lebih efektif.
  - Pipeline disamakan dengan algoritma machine learning yang lain yaitu menggunakan TransformedTargetRegressor (log_transform target).
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler()
  - Best Params XGBoost yang ditemukan adalah :
    1. n_estimators      : 200,
    2. max_depth         : 10,
    3. learning_rate     : 0.05
  - Kinerja :
    - R²  : 0.9552
    - MAE : 2761.84
    - RMSE : 4803.60

8. 🌿 LightBGM Regressor (Light Gradient Boosting Machine)
  - LightBGM adalah kerangka kerja *Gradient Bossting* yang dikembangkan oleh Microsoft yang dikenal karena efisiensi tinggi, kecepatan pelatihan yang luar biasa dan penggunaan memori yang rendah. Tujuan utama desainnya adalah untuk secara efektif menangani dataset besar dan dimensi tinggi di mana implementasi boosting tradisional mungkin melambat atau membutuhkan sumber daya komputasi yang besar. Inti dari kecepatan LightBGM adalah mekanisme pertumbuhan pohonnya yang berbeda yang disebut strategi Leaf-wise (berbasis daun). LightBGM mengandalkan dua teknik inovatif untuk mengoptimalkan efisiensi komputasi dan memori yaitu **Gradient-based One-side Sampling (GOSS)**, teknik bertujuan mengurangi jumlah sampel data pelatihan secara cerdas dan **Exclusive Feature Bundilng (EFB)** teknik yang bertujuan untuk mengurangi dimensi fitur pada dataset yang sparse (jarang, banyak nol). Secara keseluruhan, LightGBM adalah implementasi *Gradient Boosting* yang sangat canggih yang dioptimalkan untuk skalabilitas dan kecepatan menjadikannya alat pilihan bagi data scientiest yang bekerja dengan volume data yang besar.
  - Pipeline disamakan dengan algoritma machine learning yang lain yaitu menggunakan TransformedTargetRegressor (log_transform target).
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler()
  - Best Params LightGBM yang ditemukan adalah :
    1. n_estimators      : 500,
    2. max_depth         : 20,
    3. learning_rate     : 0.2
  - Kinerja :
    - R²  : 0.9552
    - MAE : 2809.01
    - RMSE : 4861.22

9. 🐱 CatBoost Regressor

  - CatBoost (Categorical Boosting) adalah algoritma **gradient boosting** berbasis pohon keputusan yang dikembangkan oleh Yandex. CatBoost dirancang untuk menangani data tabular secara efisien terutama yang mengandung banyak fitur kategorikal tanpa perlu preprocessing kompleks seperti one-hot encoding. Cara kerja singkat CatBoost:
    1. CatBoost membangun model berbasis gradient boosting seperti XGBoost dan LightGBM
    2. Algoritma ini menambahkan tree baru secara iteratif untuk memperbaiki error dari prediksi sebelumnya
    3. CatBoost memiliki algoritma *Ordered Boosting* untuk mencegah overfitting dan target leakage
    4. Model mengoptimalkan *loss function* (misalnya RMSE pada regresi) sampai mencapai jumlah iterasi maksimum atau early stopping
  -  Pipeline disamakan dengan algoritma machine learning yang lain yaitu menggunakan TransformedTargetRegressor (log_transform target).
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler()
  - Best Params CatBoost yang ditemukan adalah :
    1. iterations        : 1500,
    2. max_depth         : 10,
    3. learning_rate     : 0.15
    4. l2_leaf_reg       : 15
  - Kinerja :
    - R²  : 0.9553
    - MAE : 2737.66
    - RMSE : 4801.70

10. ⚔️ Support Vector Machine Regressor (Training no completed 🚫)
  - Support Vector Machine (SVM) adalah algoritma supervised learning yang inti dari SVM adalah menemukan batas keputusan *(Decision Boundary)* terbaik, yang disebut **Hyperplane** yang secara optimal memisahkan kelas-kelas data dalam ruang fitur. Konsep utama adalah sebagai berikut:
    1. Hyperplane, sebuah batas keputusan yang memisahkan kelas-kelas data. Dalam ruang dua dimensi, hyperplane adalah sebuah garis. Dalam ruang tiga dimensi, hyperplane adalah sebuah bidang datar. Dalam dimensi yang lebih tinggi (N-dimensi), hyperplane adalah objek geometris (N-1) dimensi.
    2. Margin, jarak antara hyperplane dan titik data terdekat dari setiap kelas.
    3. Support Vector, Titik-titik data (vektor) yang terletak paling dekat dengan hyperplane dan secara langsung mempengaruhi posisi orientasi hyperplane. Titik-titik ini adalah yang mendukung hyperplane maka dari itu nama algoritma ini.
  - Tujuan utama SVM adalah mencari hyperplane dengan margin terbesar (maximum margin), karena margin yang lebih besar cenderung menghasilkan kemampuan generalisasi yang lebih baik dan mengurangi risiko overfitting.
  -  Pipeline disamakan dengan algoritma machine learning yang lain yaitu menggunakan TransformedTargetRegressor (log_transform target).
  - Menerapkan logaritma natural dari nilai +1 lalu distandarisasi oleh StandardScaler().
  - Eksperimen ini menggunakan SVM dengan kernel RBF untuk memprediksi harga tiket pesawat. Model ini berjalan selama lebih dari 11 jam tanpa menyelesaikan proses pelatihan karena kompleksitas kernel non-linear pada dataset berskala besar.
  - Hasil ini memberikan wawasan penting bahwa untuk dataset besar, model berbasis **gradient bossting** lebih efisien dan praktiks.

## 🛠️ Cara Menggunakan Model
Contoh untuk Random Forest:
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
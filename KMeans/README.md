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


---

## 🚀 Fitur Utama

### 1. Arsitektur Pipeline
- Preprocessing seperti *StandardScaler* dan model dibungkus dalam `Pipeline`.
- Menjamin data latih dan data baru diproses dengan split lalu melakukan preprocessing hanya dengan data latih, tujuannya adalah untuk mencegah *data leakage*.

### 3. Preprocessing Fitur (KRITIS)
- Preprocessing dilakukan hanya pada data latih yang telah di split sebelumnya agar menghindari kebocoran data.
- Seperti yang dibahas sebelumnya, K-Means menggunakan jarak *Euclidean*. Oleh karena itu, *StandardScaler* atau *Min-Max Scaler* wajib digunakan untuk memastikan semua fitur berkontribusi secara adil pada perhitungan jarak, menhindari dominasi fitur dengan skala yang lebih besar.
- Preprocessing yang dilakukan hanyalah menggunakan *StandardScaler*, serta menghapus nilai negatif pada nilai feature
- Penghapusan nilai negatif tidak dapat dibungkus dengan `Pipeline`. Alasan utamanya adalah karena operasi tersebut bukanlah sebuah transformasi data standar dan biasanya melanggar aturan inti dari desain *machine learning* yang baik. Penghapusan kolom maupun nilai negatif bukan transformasi data, maka dari itu tidak dapat dibungkus dengan `Pipeline`.

### 4. Building Model K-Means
Membangun model K-Means dilakukan dengan menentukan parameter yang diperlukan ada di K-Means. Berikut parameter yang dibuat serta diinisiasi:
1. n_clusters = 2
  - Jumlah cluster (K) yang diinginkan. Disini saya ingin membagi hanya 2 cluster yang ingin saya ketahui dari datasheet traffic IoT ini. Hal itu adalah *anomaly* dan normal traffic. Algoritma ini akan mencoba membagi data menjadi tepat 2 kelompok. Pemilihan nilai K ini sangat mempengaruhi hasil clustering.
2. init = 'k-means++
  - Metode inisialisasi centroid awal menggunakan k-means++ adalah praktik terbaik karena ia memilih centeroid awal yang sudah terpisah jauh satu sama lain. Tujuannya adalah mempercepat konvergensi dan menghindari hasil clustering yang buruk (local optima) yang mungkin terjadi jika centeroid dipilih secara acak.
3. n_init = 50
  - Jumlah percobaan inisialisasi yang berbeda. Menentukan berapa kali algoritma K-Means akan dijalankan dengan *centroid* awal yang berbeda-beda. Setelah 50 kali percobaan, *Scikit-learn* akan memilih hasil terbaik (yaitu, hasil dengan *Within-Cluster Sum of Squares (WCSS) terkecil). Nilai yang lebih tinggi meningkatkan peluan menemukan solusi global yang optimal.
4. max_iter = 500
  - Jumlah maksimum iterasi per percobaan. Batas atas berapa kali langkah penugasan *cluster* dan pembaruan *centroid* akan diulang dalam satu kali run K-Means. Proses akan berhenti jika konvergen (tidak ada perubahan *centroid*) atau setelah mencapai 500 iterasi.
5. tol = 1e-5
  - Toleransi konvergensi. Batasan numerik yang digunakan untuk menentukan kapan algoritma dianggap telah konvergen (berhenti). Jika perubahan WCSS antara dua iterasi berturut-turut kurang dari 10^-5, algoritma berhenti karena dianggap telah mencapai solusi stabil.
6. algorithm = 'elkan'
  - Algoritma yang digunakan untuk penghitungan jarak. `elkan` adalah varian yang lebih efisien dari algoritma K-Means standar. Ini menggunakan ketidaksetaraan segitiga *(triangle inequality)* untuk menghindari perhitungan jarak Euclidean yang tidak perlu, sehingga dapat mempercepat proses *clustering*, terutama pada data dengan dimensi tinggi.
7. random_state = 42
  - Seed untuk generator angka acak. Digunakan untuk memastikan hasil yang dapat direplikasi *(reproducible)*. Karena inisialisasi *centroid* melibatkan keacakan (meskipun menggunakan k-means++), menetapkan random_state memastikan bahwa akan mendapatkan hasil yang sama setiap kali anda menjalankan kode dengan paramter yang sama.

Dengan pengaturan ini, saya telah mengoptimalkan K-Means untuk kecepatan dan kualitas hasil:
1. Kualitas: penggunaan init='k-means++' dan n_init=50 sangat meningkatkan peluan untuk mendapatkan clustering yang optimal.
2. Kecepatan: penggunaan *algorithm*= 'elkan' akan mempercepat waktu komputasi dibandingkan algoritma K-Means standar, terutama pada dataset yang besar.
3. Stabilitas: penggunaan random_state=42 menjamin bahwa setiap kali melatih model ini, hasilnya akan sama yang krusial untuk pengujian dan perbandingan.


**Parameter Penting pada K-Means()**
|Parameter                |  Fungsi                               | Dampak                              |
|-------------------------|---------------------------------------|-------------------------------------|
|n_clusters             | Jumlah cluster (K) yang ingin dibentuk | Parameter yang mempengaruhi klastering. Ditentukan dengan Elbow, Silhouette dsb|
|init                   | Metode inisialisasi centroid awal     | `k-means++` lebih stabil dan cepat; menghindari centroid awal buruk|
|max_iter       | Iterasi maksimum per 1 run K-Means      | Default 300; iterasi berhenti jika sudah konvergen|
|n_init         | Berapa kali K-Means diulang dengan centroid berbeda| Menghindari local optimum; memilih hasil terbaik|
|tol            | Batas toleransi untuk konvergensi     | Semakin kecil lebih presisi tapi lebih lambat|
|algorithm      | Algoritma perhitungan (`lloyd` atau `elkan`)|`elkan` lebih cepat pada data convex & rendah dimensi|
|random_state   | Seed untuk random number generator  | Agar hasil konsisten dan reproducible|
|verbose (opsional)| Menampilkan proses selama training| Berguna untuk debugging|
|copy_x| Apakah data  X disalin sebelum diproses  | Biasanya tidak perlu diubah; default True|

---

## 📈 Hasil Kinerja (Data Uji)

Menilai model menggunakan 3 metrik utama dalam proyek repository ini, metrik yang digunakan yaitu:
1. Silhoutte Score, mengukur seberapa baik objek berada di dalam clusternya dibandingkan cluster lain.
2. Davies-Bouldin Score (DBI), mengukur rata-rata kesamaan antar cluster. DBI memperhitungkan jarak antar centroid dan sebaran tiap cluster. Mirip kebalikan dari silhoutte, semakin kecil semakin baik.
3. Pairwise_distances_argmin_min, ini bukan skor tapi fungsi untuk menghitung jarak tiap titik ke centroid terdekat. Biasanya mengembalikan indices -> index cluster terdekat dan distances -> jarak minimum tiap titik ke centroidnya. Memiliki kegunaan untuk melihat seberapa dekat titik ke centroid cluster menggunakan compactness. Bisa dipakai untuk mencari outlier (distance besar) visualisasi kualitas cluster dan menghitung averange distance to centroid.

| Metrik | Nilai |
|--------|-------|
| **Silhoutte Score** | 0.849 | 
| **Bouldin_Score** | 0.503|
| **Pairwise Distances_argmin_min** | array([0.246, 0.045]) |

**Interpretasi Angka**
1. Silhouette score = 0.849,
  - Nilai 0.849 mengindikasikan bahwa struktur klaster yang terbentuk sangat kuat dan terpisah dengan jelas.
  - Secara teoritis, nilai silhouette di atas 0.80 merupakan indikasi bahwa setiap objek memiliki tingkat kesesuaian yang tinggi terhadap klusternya sendiri dan memiliki jarak yang signifikan dari kluster lainnya.
  - Dengan demikian, klaster yang dihasilkan memiliki kohesi internal yang kuat *(intra-cluster cohension)* serta separasi antar klaster yang optimal *(inter-cluster separation)*.
  - Dalam konteks data ini yang memiliki +5 juta *rows* berukuran sangat besar, skor setinggi ini menegaskan bahwa model berhasil menemukan batas klaster yang secara alami memang terbentuk pada data tersebut.

2. Davies-Bouldin Score = 0.503,
  - Semakin memperkuat temuan bahwa kualitas model berada pada kategori sangat baik.
  - Davies Bouldin index mengukur seberapa mirip suatu cluster dengan cluster lain berdasarkan rasio antara sebaran internal dan jarak antar centroid.
  - Nilai DBI yang mendekati nol mencerminkan bahwa setiap cluster memiliki disperensi internal yang relatif kecil sekaligus berada pada jarak yang cukup jauh dari cluster lainnya.
  - Nilai 0.503 tergolong rendah dan menunjukkan bahwa bahwa struktur kluster yang dihasilkan efisien, tidak saling tumpang tindih dan centroid mampu mempresentasikan anggota klusternya dengan baik.

3. pairwise_distances_argmin_min = ([0.246, 0.045])
  - Nilai tersebut adalah jarak centroid yang memberikan informasi tentang tingkat kekompakan masing-masing cluster.
  - Rata-rata jarak titik terhadap centroid sebesar [0.246, 0.045] menunjukkan kedua cluster padat dengan cluster 1 memiliki struktur yang paling kompak.
  - Cluster 1 = 0.045, merupakan cluster yang sangat padat *(super tight)*
  - Cluster 0 = 0.246, masih padat, tapi sedikit lebih menyebar dibanding cluster 1.
  - Cluster dengan nilai rata-rata jarak 0.045 menunjukkan kompaktitas yang sangat tingi, menandakan bahwa anggota cluster tersebut hampir seluruhnya terkonsentrasi dekat dengan centroid.
  - Ini merupakan indikator cluster yang sangat homogen
  - Cluster dengan jarak rata-rata 0.246 masih tergolong baik, namun menunjukkan bahwa cluster tersebut memiliki dispersi sedikit lebih besar dibandingkan cluster pertama yang dapat disebabkan oleh variasi data yang lebih tinggi atau distribusi yang lebih lebar. Perbedaan ini tidak mengindikasikan kelemahan, melainkan karakteristik alami dari data yang tergabung pada cluster tersebut.

Secara keseluruhan, kombinasi ketiga metrik ini memberikan gambaran yang konsisten bahwa model K-Means kamu tidak hanya stabil, tetapi juga mampu membentuk klaster dengan kualitas yang sangat tinggi meskipun diterapkan pada dataset berukuran lebih dari 5 juta entri. Hasil ini menunjukkan bahwa pemilihan jumlah cluster yang kamu gunakan tepat, inisialisasi centroid efektif dan struktur alaminya memang mendukung pembentukan klaster yang jelas. Temuan ini bisa menjadi poin kuat dalam analisis akademik maupun penulisan ilmiah karena mencerminkan performa model yang robust pada skala data besar.

---

## 📊 Visualisasi Data
### 1. PCA 2D Clustering
![PCA 2D](Assets/PCA%202D%20Clustering.png)<br>
PCA (Principal Component Analysis) adalah teknik reduksi dimensi yang mengubah data berdimensi tinggi menjadi beberapa komponene baru yang saling ortogonal (tidak berkorelasi), menyimpan variasi paling besar pada data dan memampatkan informasi data ke dalam dimensi lebih sedikit tanpa kehilangan struktur penting. Pada visualisasi ini 2D berarti mengompresi data menjadi 2 komponen utama PC1 -> menjelaskan variasi terbesar dan PC2 -> menjelaskan variasi terbesar kedua. Fungsi utamanya adalah untuk visualisasi cluster, melihat separasi antar kelas, memeriksa K-Means memilih cluster yang benar dan menemukan pola yang tidak terlihat dalam data asli

### 2. PCA 3D Clustering
![PCA 3D Clustering](Assets/PCA%203D%20Clustering.png)<br>
PCA 3D sama persis idenya, tetapi ini menyimpan tiga komponen utama yaitu PC1, PC2 dan PC3. Setiap titik data sekarang memiliki koordinat, fungsi utamanya melakukan visualisasi 3D untuk dataset yang kompleks dan PCA 3D membantu saat. Data terlalu kompleks untuk terlihat di 2D, cluster saling tumpang tindih jika hanya dilihat dari 2 komponen dan banyak variasi data muncul pada komponen ke 3.

### 3. Jumlah Data Percluster
![Jumlah data percluster](Assets/Jumlah%20Data%20PerCluster.png)<br>
Visualisasi ini menampilkan jumlah data percluster, dapat dilihat pada cluster 0 memiliki dominan hampir 99% data yang berada di cluster 0, sedangkan cluster 1 hanya 1% dari jumlah data. Menunjukkan cluster 0 sangat dominan dalam datasheet 5 juta rows ini.
 
### 5. Cluster 0 vs 1 Berdasarkan pktrate
![Perbandingan cluster 0 vs 1 berdasarkan pktrate](Assets/Cluster%200%20vs%201%20base%20features%20pktrate.png)<br>
Visualisasi ini menampilkan boxplot cluster 0 vs cluster 1, dimana nilai median pada cluster 0 lebih rendah dibanding cluster 1. Cluster 0 menunjukkan box yang lebih besar menandakan banyak variasi di dalamnya, cluster 0 memiliki median di angka rentang antara 0.00010 - 0.00015. Ini seperti ciri-ciri dari traffic normal/stabil. Sedangkan cluster 1 memiliki nilai median di rentang sekitar 0.00020, nilai Q3 berada paling atas dibanding nilai Q1 pada cluster 0.

## 6. Distribusi `dur` Berdasarkan Cluster
![Distribusi fitur dur berdasarkan cluster](Assets/Distribusi%20dur%20berdasarkan%20cluster.png)

## 7. Distribusi `pktrate` Berdasarkan Cluster
![Distribusi pktrate](Assets/Distribusi%20pktrate%20berdasarkan%20cluster.png)

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
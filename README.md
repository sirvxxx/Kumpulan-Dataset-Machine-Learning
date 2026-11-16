# Kumpulan Dataset & Latihan Machine Learning

Repositori ini berisi kumpulan dataset beserta implementasi praktis (*hands-on*) berbagai algoritma **Machine Learning** menggunakan Python, NumPy, Pandas, Matplotlib, dan Scikit-Learn.

---

## 📁 Struktur & Daftar Proyek

| No | Judul Latihan / Modul | Dataset | Algoritma / Topik | File Notebook |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Linear SVM dari Nol & Scikit-Learn** | Studi Kasus Sintetis (Lama Belajar & Lama Tidur) | Support Vector Machine (`SVC` Kernel Linear & Manual NumPy) | `latihan_svm_linear.ipynb` |
| 2 | **Klasifikasi Diabetes dengan SKLearn SVM** | Pima Indians Diabetes (`diabetes.csv`) | Support Vector Classification (`SVC` Kernel RBF/Default) | `latihan_sklearn_svm_klasifikasi.ipynb` |
| 3 | *(Tambahkan dataset lain di repo Anda di sini)* | `nama_dataset.csv` | *(Algoritma yang digunakan)* | `nama_notebook.ipynb` |

---

## 🧠 Penjelasan Pengerjaan Masing-Masing Modul

### 1. Latihan SVM — Linear Kernel (`latihan_svm_linear.ipynb`)
Modul ini berfokus pada pemahaman konsep matematis dan geometris dari algoritma **Support Vector Machine (SVM)** pada kasus pemisahan dua kelas secara linear (*Linearly Separable*).

* **Dataset:** Data sintetis 2 dimensi yang merepresentasikan **Lama Belajar ($X_1$)** dan **Lama Tidur ($X_2$)** terhadap **Status Kelulusan ($Y$)** (`Lulus` vs `Tidak Lulus`).
* **Tahapan Pengerjaan:**
  1. **Import Library:** Menggunakan `numpy` untuk komputasi matriks, `matplotlib.pyplot` untuk visualisasi grafik, dan `sklearn.svm.SVC` untuk pemodelan SVM.
  2. **Persiapan Data:** Menyusun fitur `X1` dan `X2` menjadi matriks fitur 2D berukuran `(8, 2)` serta mendefinisikan label target biner (`+1`/`-1` untuk versi manual dan `1`/`0` untuk Scikit-Learn).
  3. **Pelatihan Model (Dua Pendekatan):**
     * **Manual dengan NumPy:** Mengimplementasikan algoritma *Gradient Descent* dengan *Hinge Loss* dan regularisasi ($C = 10$) untuk mencari bobot optimal ($w_1, w_2$) serta bias ($b$) secara iteratif.
     * **Scikit-Learn (`SVC`):** Melatih model menggunakan `SVC(kernel='linear', C=10)`, mengekstrak parameter *hyperplane* (`coef_` dan `intercept_`), mengidentifikasi titik *support vectors*, serta menguji prediksi pada data baru.
  4. **Visualisasi Decision Boundary & Margin:** Memplot sebaran titik data kedua kelas, lingkaran penanda *Support Vectors*, garis *hyperplane* ($w \cdot x + b = 0$), serta garis *Upper Margin* ($+1$) dan *Lower Margin* ($-1$).

---

### 2. Latihan SKLearn SVM untuk Klasifikasi (`latihan_sklearn_svm_klasifikasi.ipynb`)
Modul ini mengimplementasikan alur kerja (*pipeline*) klasifikasi Machine Learning standar menggunakan dataset medis dunia nyata untuk memprediksi risiko diabetes pada pasien.

* **Dataset:** **Pima Indians Diabetes** (`diabetes.csv` / `diabetes_2.csv`) yang terdiri dari **768 sampel** dan **8 fitur klinis** (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) dengan target `Outcome` (`0` = Negatif, `1` = Positif).
* **Tahapan Pengerjaan:**
  1. **Memuat & Eksplorasi Data (`Pandas`):** Membaca file CSV ke dalam DataFrame (`pd.read_csv`), meninjau sampel data awal (`df.head()`), serta memverifikasi kelengkapan data (*non-null*) dan tipe data numerik (`df.info()`).
  2. **Pemisahan Atribut & Label:** Memisahkan 8 kolom pertama sebagai matriks fitur prediktor (`X`) dan kolom `Outcome` sebagai target klasifikasi (`y`).
  3. **Standardisasi Fitur (`StandardScaler`):** Menyamakan skala distribusi seluruh fitur (rata-rata = 0, standar deviasi = 1) agar fitur dengan rentang angka besar (seperti `Insulin`) tidak mendominasi perhitungan jarak pada algoritma SVM.
  4. **Pembagian Data Latih & Data Uji (`train_test_split`):** Membagi dataset dengan rasio **67% data latih** (`X_train`, `y_train`) dan **33% data uji** (`X_test`, `y_test`) menggunakan `random_state=42` agar hasil eksperimen konsisten.
  5. **Pembuatan & Pelatihan Model (`SVC`):** Menginisialisasi objek `SVC()` dengan konfigurasi default dan melatihnya pada data latih menggunakan `.fit(X_train, y_train)`.
  6. **Evaluasi Model:** Mengukur performa klasifikasi pada data uji menggunakan `clf.score(X_test, y_test)` yang menghasilkan tingkat akurasi sekitar **75,59%** (`0.7559`).

---

### 3. Dataset Lainnya di Repositori Ini *(Silakan Sesuaikan)*
> *Catatan: Karena repositori GitHub Anda bersifat private / belum terindeks publik, Anda dapat menyalin format di bawah ini untuk mendokumentasikan file dataset lain yang sudah ada di repositori Anda:*

* **Dataset:** `[Nama Dataset, misal: Iris / Titanic / Housing]` (`nama_file.csv`)
* **Tujuan:** `[Klasifikasi / Regresi / Clustering]`
* **Tahapan Pengerjaan:**
  1. **Preprocessing:** Penanganan *missing values*, encoding data kategorikal, dan normalisasi/standardisasi.
  2. **Splitting:** Pembagian data latih (*training*) dan data uji (*testing*).
  3. **Modeling & Evaluasi:** Pelatihan model menggunakan `[Nama Algoritma]` dan evaluasi menggunakan metrik `[Akurasi / MSE / R2 Score]`.

---

## 🚀 Cara Menjalankan Notebook

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/sirvxxx/Kumpulan-Dataset-Machine-Learning.git
   cd Kumpulan-Dataset-Machine-Learning
   ```
2. **Instalasi dependensi yang diperlukan:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```
3. **Jalankan Jupyter Notebook atau buka langsung di Google Colab:**
   ```bash
   jupyter notebook
   ```
   *(Pastikan file dataset seperti `diabetes.csv` berada di direktori yang sama dengan file `.ipynb` saat menjalankan kode).*
README.txt - Panduan Penggunaan Model Estimasi Nutrisi Makanan
Updated: 10/06/2025

1. Deskripsi Singkat
Sistem ini adalah pipeline machine learning yang dirancang untuk menganalisis gambar
makanan dan memberikan estimasi nutrisi lengkap. Prosesnya melibatkan dua model utama:

Model Deteksi & Segmentasi (YOLO): Untuk mengidentifikasi jenis makanan dan areanya dalam gambar.
Model Estimasi Nutrisi (SVR): Untuk memprediksi kandungan nutrisi berdasarkan jenis dan estimasi berat makanan.
Tujuan akhirnya adalah mengubah sebuah gambar makanan menjadi output data nutrisi (karbohidrat, protein, lemak, dan kalori).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2. File Aset yang Diperlukan
Untuk menjalankan sistem ini secara penuh, pastikan semua file aset berikut tersedia dan berada di path yang benar:

- Model Deteksi (YOLO):
    model_yolov11-seg.pt

- Model Estimasi Nutrisi (SVR) & Komponennya:
    svr_multi_output_model.pkl
    svr_scaler.pkl
    svr_feature_names.pkl

- Data Pendukung:
    pixel_dataset.csv (berisi faktor konversi piksel ke gram)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3. Alur Kerja Estimasi (Dari Gambar ke Nutrisi)
Berikut adalah alur kerja langkah demi langkah tentang bagaimana sistem memproses input gambar untuk
menghasilkan output nutrisi. Referensi alur kerja ini dapat dilihat di file model-run-complete.ipynb.

Input: Sebuah file gambar makanan (misalnya, dalam format .jpg atau .png).
Output: Rincian estimasi nutrisi untuk setiap makanan yang terdeteksi di gambar.

--- Langkah-langkah Proses ---

> Preprocessing Gambar:

    Aplikasi menerima gambar makanan sebagai input.
    Untuk hasil terbaik, disarankan agar gambar diambil dari atas (top-down view) dengan jarak yang relatif konsisten (~30cm).
    Gambar input kemudian diubah ukurannya (resize) menjadi 640x640 piksel.

> Tahap 1: Deteksi & Estimasi Berat (Menggunakan YOLO)

    Gambar 640x640 dimasukkan ke Model YOLO (model_yolov11-seg.pt).
    Model akan melakukan instance segmentation untuk menghasilkan dua informasi utama untuk setiap objek makanan yang terdeteksi:
    Jenis Makanan (food_type): Contoh: 'Nasi Putih', 'Ayam Goreng - Dada'.
    Jumlah Piksel (pixel_count): Luas area objek dalam piksel dari masker segmentasi.
    Untuk setiap makanan yang terdeteksi, estimasi berat (gram) dihitung dengan mengalikan pixel_count dengan
    faktor gram_per_pixel yang sesuai dari file pixel_dataset.csv.

> Tahap 2: Estimasi Nutrisi (Menggunakan SVR)

    Hasil dari Tahap 1, yaitu food_type dan estimasi_berat, digunakan sebagai input untuk tahap ini.
    Preprocessing Input SVR: Sebelum dimasukkan ke model SVR, input ini diproses menggunakan:
    svr_feature_names.pkl: Untuk memastikan struktur fitur (setelah one-hot encoding) konsisten.
    svr_scaler.pkl: Untuk melakukan scaling pada data input.
    
    Prediksi Nutrisi: Data yang telah diproses kemudian dimasukkan ke Model SVR (svr_multi_output_model.pkl).
    Model SVR akan mengeluarkan estimasi nutrisi lengkap untuk makanan tersebut:
    Karbohidrat (g)
    Protein (g)
    Lemak (g)
    Kalori (kkal)
    Agregasi Hasil:

    Jika ada beberapa jenis makanan yang terdeteksi di gambar, proses ini akan diulang untuk setiap jenis.
    Sistem dapat menampilkan rincian nutrisi per makanan dan juga total nutrisi keseluruhan dari piring tersebut

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4. Dependensi Library Python
Untuk menjalankan skrip inferensi (model-run-complete.ipynb atau sejenisnya), pastikan environment Python memiliki library berikut terinstal:

    pandas
    numpy
    opencv-python (untuk cv2)
    torch & torchvision
    ultralytics (untuk model YOLO)
    scikit-learn
    joblib

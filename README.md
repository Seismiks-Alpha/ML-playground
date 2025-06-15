README.txt - Panduan Penggunaan Model Estimasi Nutrisi Makanan
Updated: 10/06/2025

1. Deskripsi Singkat
Sistem ini adalah pipeline machine learning yang dirancang untuk menganalisis gambar
makanan dan memberikan estimasi nutrisi lengkap. Prosesnya melibatkan dua model utama:

Model Deteksi & Segmentasi (YOLO): Untuk mengidentifikasi jenis makanan dan areanya dalam gambar.
Model Estimasi Nutrisi (SVR): Untuk memprediksi kandungan nutrisi berdasarkan jenis dan estimasi berat makanan.
Tujuan akhirnya adalah mengubah sebuah gambar makanan menjadi output data nutrisi (karbohidrat, protein, lemak, dan kalori).

---------------------------------------------------------------------------------------------------------------------------------

2. File Aset yang Diperlukan
Untuk menjalankan sistem ini secara penuh, pastikan semua file aset berikut tersedia dan berada di path yang benar:

- Model Deteksi (YOLO):
    model_yolov11-seg.pt

- Model Klasifikasi Volume (CNN):
    volume_classifier_finetuned_model.h5

- Model Estimasi Nutrisi (SVR) & Komponennya:
    svr_multi_output_model.pkl
    svr_scaler.pkl
    svr_feature_names.pkl

- Data Pendukung:
    pixel_dataset.csv (berisi faktor konversi piksel ke gram)

---------------------------------------------------------------------------------------------------------------------------------

3. Alur Kerja Estimasi (Dari Gambar ke Nutrisi)
Input: Sebuah file gambar makanan.
Output: Rincian estimasi nutrisi untuk setiap makanan yang terdeteksi.

Langkah-langkah Proses:

Pra-pemrosesan Gambar: Gambar input diubah ukurannya menjadi 640x640 piksel.

Deteksi & Segmentasi (YOLO): Model YOLO mendeteksi food_type, pixel_count (luas area), dan bounding_box untuk setiap objek.

Klasifikasi Volume (CNN - Khusus Nasi Putih): Jika objek yang terdeteksi adalah 'Nasi Putih', gambar potongan (crop) berdasarkan bounding_box akan dianalisis oleh Model CNN untuk menentukan volumenya ('merata' atau 'padat').

Estimasi Berat: Berat diestimasi dengan mengalikan pixel_count dengan faktor gram_per_pixel dari pixel_dataset.csv. Untuk Nasi Putih, faktor yang digunakan disesuaikan berdasarkan hasil klasifikasi CNN.

Estimasi Nutrisi (SVR): Estimasi berat dan nama makanan dasar (misalnya, 'Nasi Putih', tanpa label 'merata'/'padat') digunakan sebagai input untuk Model SVR untuk memprediksi nutrisi lengkap.

---------------------------------------------------------------------------------------------------------------------------------

4. Dependensi Library Python
Untuk menjalankan skrip inferensi (model-run-complete.ipynb atau sejenisnya), pastikan environment Python memiliki library berikut terinstal:

    pandas
    numpy
    opencv-python (untuk cv2)
    torch & torchvision
    ultralytics (untuk model YOLO)
    scikit-learn
    joblib

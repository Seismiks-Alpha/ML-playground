# ==============================================================================
# SEISMIKS ALPHA - COMPLETE INFERENCE PIPELINE
# ==============================================================================

# --- 1. Impor Library yang Dibutuhkan ---
print("--- Mengimpor library... ---")
import os
import cv2
import torch
import pandas as pd
import joblib
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# ==============================================================================
# BAGIAN I: MEMUAT SEMUA MODEL DAN DATA PENDUKUNG
# ==============================================================================
print("\n--- Memuat semua model dan data aset... ---")

# --- Muat Model YOLO ---
try:
    # Path disesuaikan untuk membaca dari folder yang sama
    yolo_model = YOLO("model_yolov11-seg.pt") 
    print("Model YOLO berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model YOLO: {e}")
    exit()

# --- Muat Faktor Konversi gram_per_pixel ---
try:
    conversion_factors_df = pd.read_csv('pixel_dataset.csv')
    gram_per_pixel_map = pd.Series(conversion_factors_df.gram_per_pixel.values,
                                   index=conversion_factors_df.food_type).to_dict()
    print("Data faktor konversi 'pixel_dataset.csv' berhasil dimuat.")
except Exception as e:
    print(f"Error memuat 'pixel_dataset.csv': {e}")
    gram_per_pixel_map = {}

# --- Muat Model CNN Klasifikasi Volume ---
try:
    # Path disesuaikan, pastikan nama file ini benar
    cnn_volume_model = tf.keras.models.load_model('CNN_model.h5') 
    cnn_class_names = ['merata', 'padat'] 
    print("Model CNN klasifikasi volume berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model CNN: {e}")
    cnn_volume_model = None

# --- Muat Model SVR Estimasi Nutrisi ---
try:
    svr_nutrition_model = joblib.load('svr_multi_output_model.pkl')
    svr_scaler = joblib.load('svr_scaler.pkl')
    svr_feature_names = joblib.load('svr_feature_names.pkl')
    print("Model SVR estimasi nutrisi berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model SVR: {e}")
    svr_nutrition_model = None

# ==============================================================================
# BAGIAN II: PROSES GAMBAR DAN EKSEKUSI PIPELINE
# ==============================================================================

def get_full_food_analysis(image_path):
    """
    Fungsi utama untuk menjalankan seluruh pipeline analisis makanan dari gambar.
    """
    # --- A. Pra-pemrosesan Gambar Input ---
    print(f"\n--- Memproses gambar: {os.path.basename(image_path)} ---")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Gagal membaca gambar dari {image_path}")
        return None
    
    target_size = (640, 640)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # --- B. Deteksi dan Segmentasi dengan YOLO ---
    yolo_results = yolo_model(source=resized_image, verbose=False)
    print("YOLO: Deteksi dan segmentasi selesai.")

    # --- C. Ekstraksi Informasi dari Hasil YOLO ---
    detected_objects = [] 
    
    if yolo_results[0].masks is not None and hasattr(yolo_results[0].boxes, 'cls'):
        for i in range(len(yolo_results[0].masks.data)):
            class_id = int(yolo_results[0].boxes.cls[i].item())
            class_name = yolo_results[0].names[class_id]
            pixel_count = torch.sum(yolo_results[0].masks.data[i]).item()
            bounding_box = yolo_results[0].boxes.xyxy[i].cpu().numpy().astype(int)
            
            lookup_food_type = class_name
            
            # --- D. [Kondisional] Klasifikasi Volume dengan CNN ---
            if class_name == 'Nasi Putih' and cnn_volume_model is not None:
                print(f"YOLO mendeteksi '{class_name}'. Menjalankan model CNN untuk klasifikasi volume...")
                
                x1, y1, x2, y2 = bounding_box
                cropped_img = resized_image[y1:y2, x1:x2]
                
                if cropped_img.size > 0:
                    cnn_input_img = cv2.resize(cropped_img, (224, 224))
                    cnn_input_img = np.expand_dims(cnn_input_img, axis=0)
                    cnn_input_img = tf.keras.applications.mobilenet_v2.preprocess_input(cnn_input_img)
                    
                    prediction = cnn_volume_model.predict(cnn_input_img, verbose=0)
                    volume_label = cnn_class_names[int(prediction[0][0] > 0.5)]
                    
                    lookup_food_type = f"{class_name} - {volume_label.capitalize()}"
                    print(f"CNN: Klasifikasi volume untuk '{class_name}' adalah -> '{volume_label}'")
            
            detected_objects.append({'food_type': lookup_food_type, 'pixel_count': pixel_count})
    
    # --- E. Agregasi Piksel untuk Makanan Sejenis ---
    aggregated_pixels = {}
    for obj in detected_objects:
        aggregated_pixels[obj['food_type']] = aggregated_pixels.get(obj['food_type'], 0) + obj['pixel_count']
        
    # --- F. Estimasi Berat ---
    estimated_weights = {}
    if gram_per_pixel_map:
        for lookup_name, total_pixels in aggregated_pixels.items():
            if lookup_name in gram_per_pixel_map:
                g_per_pixel = gram_per_pixel_map[lookup_name]
                estimated_weight = total_pixels * g_per_pixel
                estimated_weights[lookup_name] = estimated_weight
    
    # --- G. Estimasi Nutrisi ---
    final_results = []
    if estimated_weights and svr_nutrition_model is not None:
        print("\n--- Hasil Akhir Estimasi Nutrisi ---")
        for lookup_name, weight in estimated_weights.items():
            
            # Tentukan nama yang akan digunakan untuk SVR dan untuk ditampilkan
            display_name = lookup_name.split(' - ')[0]
            
            # Untuk input SVR, gunakan nama dasar jika itu Nasi Putih
            if "Nasi Putih" in lookup_name:
                svr_input_food_type = "Nasi Putih"
            else:
                svr_input_food_type = lookup_name
            
            new_data = pd.DataFrame([{'food_type': svr_input_food_type, 'portion_size': weight}])
            new_data_encoded = pd.get_dummies(new_data)
            new_data_reindexed = new_data_encoded.reindex(columns=svr_feature_names, fill_value=0)
            new_data_scaled = svr_scaler.transform(new_data_reindexed)
            
            prediction = svr_nutrition_model.predict(new_data_scaled)
            adjusted_prediction = np.maximum(0, prediction[0])
            
            nutrition_dict = {
                'carbohydrates': round(adjusted_prediction[0], 2),
                'protein': round(adjusted_prediction[1], 2),
                'fat': round(adjusted_prediction[2], 2),
                'calories': int(round(adjusted_prediction[3]))
            }
            
            result_item = {
                'food_name': display_name,
                'estimated_weight_g': round(weight, 2),
                'nutrition': nutrition_dict
            }
            final_results.append(result_item)
            
            # Tampilkan hasil per item dengan nama dasar
            print(f"\n> {display_name} (Estimasi Berat: {weight:.2f} g)")
            for nutrient, value in nutrition_dict.items():
                unit = "g" if nutrient != "calories" else "kkal"
                print(f"  - {nutrient.capitalize()}: {value} {unit}")
    
    return final_results


# --- Contoh Penggunaan Fungsi Utama ---
if __name__ == "__main__":
    test_image_path = "run-tests/ayam-110gr_nasi-110gr_1.jpg" 
    final_analysis = get_full_food_analysis(test_image_path)

    if final_analysis:
        print("\n--- Ringkasan Analisis (Format JSON-like) ---")
        import json
        print(json.dumps(final_analysis, indent=4))
    else:
        print("\nAnalisis tidak dapat diselesaikan atau tidak ada objek terdeteksi.")
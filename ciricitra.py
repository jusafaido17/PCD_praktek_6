import cv2
import numpy as np
import math
from skimage.measure import regionprops
from skimage.measure import shannon_entropy
from scipy.stats import variation
import matplotlib.pyplot as plt

# --- KONFIGURASI FILE INPUT ---
# ❗ GANTI DENGAN NAMA FILE CITRA WARNA ANDA (e.g., apel.jpg) ❗
FILENAME = 'apples3.jpg' 

# RENTANG HSV UNTUK SEGMENTASI WARNA HIJAU (sesuaikan jika objek Anda berwarna lain)
# Hue (0-179), Saturation (0-255), Value (0-255)
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

def ekstraksi_ciri_all_in_one(filename):
    
    # 1. MEMBACA & SEGMENTASI CITRA (Mendapatkan Objek Biner)
    img_bgr = cv2.imread(filename)
    if img_bgr is None:
        print(f"❌ ERROR: Tidak dapat memuat file '{filename}'. Pastikan file ada.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Masking berdasarkan rentang warna HSV
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
    # Operasi morfologi untuk membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Cari Kontur Objek
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("❌ ERROR: Tidak ditemukan objek (kontur) dalam rentang warna yang ditentukan.")
        return

    # Ambil kontur terbesar dan buat masker biner final
    main_contour = max(contours, key=cv2.contourArea)
    mask_final = np.zeros_like(mask)
    cv2.drawContours(mask_final, [main_contour], -1, 255, cv2.FILLED)
    
    # =========================================================================
    # A. EKSTRAKSI CIRI UKURAN (Luas & Keliling) [cite: 12]
    # =========================================================================
    Area = cv2.contourArea(main_contour) # Luas (jumlah piksel) [cite: 13]
    Perimeter = cv2.arcLength(main_contour, True) # Keliling (jumlah piksel) [cite: 13]
    
    # =========================================================================
    # B. EKSTRAKSI CIRI BENTUK (Eccentricity & Metric) [cite: 5]
    # =========================================================================
    
    labeled_img = (mask_final > 0).astype(int)
    regions = regionprops(labeled_img)
    
    if regions:
        # Eccentricity (0=bulat, 1=memanjang) [cite: 8]
        Eccentricity = regions[0].eccentricity 
    else:
        Eccentricity = np.nan
    
    # Metric (M): M = (4 * pi * Area) / (Keliling^2) (0=memanjang, 1=bulat) [cite: 10]
    if Perimeter > 0:
        Metric = (4 * math.pi * Area) / (Perimeter ** 2)
    else:
        Metric = np.nan

    # =========================================================================
    # C. EKSTRAKSI CIRI GEOMETRI (Jarak Euclidean) [cite: 15]
    # =========================================================================
    
    # Hitung Centroid (Titik Pusat Massa)
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Hitung Bounding Box (untuk mendapatkan titik acuan)
    x, y, w_box, h_box = cv2.boundingRect(main_contour)
    
    # Jarak Euclidean antara Centroid (cx, cy) ke Titik Kiri Atas Bounding Box (x, y)
    # Persamaan Euclidean: Jarak = sqrt((x2-x1)^2 + (y2-y1)^2) 
    Jarak_Geometri = math.sqrt((x - cx)**2 + (y - cy)**2)
    
    # =========================================================================
    # D. EKSTRAKSI CIRI WARNA (Mean Hue, Saturation, Value) [cite: 28]
    # =========================================================================
    
    # Aplikasikan masker ke citra HSV asli
    masked_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_final)
    
    # Ekstrak nilai H, S, V yang valid (bukan nol karena masking)
    H_values = masked_hsv[:,:,0][mask_final > 0]
    S_values = masked_hsv[:,:,1][mask_final > 0]
    V_values = masked_hsv[:,:,2][mask_final > 0]
    
    # Hitung rata-rata (Mean) untuk setiap channel
    Mean_H = np.mean(H_values)
    Mean_S = np.mean(S_values)
    Mean_V = np.mean(V_values)
    
    # =========================================================================
    # E. EKSTRAKSI CIRI TEKSTUR (Statistik Orde Pertama) [cite: 21]
    # =========================================================================
    
    # Aplikasikan masker ke citra Grayscale
    masked_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask_final)
    
    # Ekstrak nilai piksel grayscale yang valid (bukan nol)
    Gray_values = masked_gray[mask_final > 0]
    
    # Ciri statistik orde pertama 
    Texture_Mean = np.mean(Gray_values)
    Texture_Variance = np.var(Gray_values)
    Texture_Entropy = shannon_entropy(Gray_values) # Menggunakan shannon_entropy (dari skimage)
    
    # 6. MENAMPILKAN HASIL

    print("\n=========================================================")
    print(f"HASIL EKSTRAKSI LIMA CIRI OBJEK PADA '{filename}'")
    print("=========================================================")
    
    print("\n[A & B] EKSTRAKSI CIRI UKURAN & BENTUK:")
    print(f"1. Luas (Area):\t\t\t{Area:.2f} piksel")
    print(f"2. Keliling (Perimeter):\t\t{Perimeter:.2f} piksel")
    print(f"3. Eccentricity:\t\t{Eccentricity:.4f} (0=Bulat, 1=Memanjang)")
    print(f"4. Metric:\t\t\t{Metric:.4f} (0=Memanjang, 1=Bulat)")
    
    print("\n[C] EKSTRAKSI CIRI GEOMETRI:")
    print(f"5. Jarak Euclidean (Centroid ke Sudut BB):\t{Jarak_Geometri:.2f} piksel")
    
    print("\n[D] EKSTRAKSI CIRI WARNA (HSV):")
    print(f"6. Mean Hue (Rata-rata Warna):\t{Mean_H:.2f}")
    print(f"7. Mean Saturation (Rata-rata Kemurnian):\t{Mean_S:.2f}")
    print(f"8. Mean Value (Rata-rata Kecerahan):\t{Mean_V:.2f}")
    
    print("\n[E] EKSTRAKSI CIRI TEKSTUR (Statistik Orde Pertama):")
    print(f"9. Mean (Rata-rata Intensitas):\t{Texture_Mean:.2f}")
    print(f"10. Variance (Variansi Intensitas):\t{Texture_Variance:.2f}")
    print(f"11. Entropy (Kekacauan Informasi):\t{Texture_Entropy:.4f}")
    print("=========================================================\n")
    
    # Visualisasi
    img_contour = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_contour, [main_contour], -1, (255, 0, 0), 2)
    cv2.circle(img_contour, (cx, cy), 5, (0, 0, 255), -1) # Centroid (Biru)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Citra Asli')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_contour)
    plt.title(f'Objek Tersegmen ({Area:.0f} piksel)')
    plt.axis('off')
    
    plt.show(block=True)
    

# --- JALANKAN PROGRAM PRAKTEK 6 ---
print(f"Memulai Ekstraksi Ciri pada '{FILENAME}'...")
ekstraksi_ciri_all_in_one(FILENAME)
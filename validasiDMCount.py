import os
import sys
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from torchvision import transforms
from glob import glob

# --- path setup ---
BASE_DIR = '/home/bagas/SurveilanceCam'
DM_COUNT_DIR = os.path.join(BASE_DIR, 'deps/DM_Count')
sys.path.insert(0, DM_COUNT_DIR)

from models import vgg19

# --- konfigurasi ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(DM_COUNT_DIR, 'pretrained_models/model_qnrf.pth')

# # arahkan ke folder test_data dataset ShanghaiTech
# DATA_DIR = os.path.join(BASE_DIR, 'dataset/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data')
# IMG_DIR = os.path.join(DATA_DIR, 'images')
# GT_DIR = os.path.join(DATA_DIR, 'ground_truth')

# arahkan ke folder test_data dataset QNRF
DATA_DIR = os.path.join(BASE_DIR, 'dataset/UCF-QNRF_ECCV18/Test')
IMG_DIR = os.path.join(DATA_DIR)
GT_DIR = os.path.join(DATA_DIR)
print(f'Menggunakan device: {device}')

# --- load model ---
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def load_gt_count(mat_path):
    """Ambil jumlah titik anotasi (ground truth count) dari file .mat QNRF."""
    mat = sio.loadmat(mat_path)
    points = mat['annPoints']   # array Nx2 (koordinat x,y) — format resmi UCF-QNRF
    return points.shape[0]

# def load_gt_count(mat_path):
#     """Ambil jumlah titik anotasi (ground truth count) dari file .mat ShanghaiTech."""
#     mat = sio.loadmat(mat_path)
#     points = mat['image_info'][0][0][0][0][0]  # array Nx2 (koordinat x,y)
#     return points.shape[0]

# def predict_count(img_path):
#     img = Image.open(img_path).convert('RGB')
#     w, h = img.size
#     new_w, new_h = (w // 8) * 8, (h // 8) * 8
#     if (new_w, new_h) != (w, h):
#         img = img.resize((new_w, new_h), Image.BILINEAR)
#     inputs = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs, _ = model(inputs)
#         return torch.sum(outputs).item()

def predict_count(img_path, max_side=1920):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # batasi sisi terpanjang agar tidak OOM di gambar QNRF yang sangat besar
    longest_side = max(w, h)
    if longest_side > max_side:
        scale = max_side / longest_side
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.BILINEAR)

    # pastikan tetap kelipatan 8
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.BILINEAR)

    inputs = transform(img).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs, _ = model(inputs)
            count = torch.sum(outputs).item()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [!] OOM di {os.path.basename(img_path)}, coba ukuran lebih kecil...')
        # fallback: perkecil lagi separuh dari max_side
        return predict_count(img_path, max_side=max_side // 2)

    return count

# --- loop semua gambar test ---
img_paths = sorted(glob(os.path.join(IMG_DIR, '*.jpg')))
errors = []
results = []

for img_path in img_paths:
    fname = os.path.basename(img_path)                        # img_0001.jpg
    gt_fname = fname.replace('.jpg', '_ann.mat')                # img_0001_ann.mat
    # gt_fname = 'GT_' + fname.replace('.jpg', '.mat')          # GT_IMG_1.mat (ShanghaiTech)
    gt_path = os.path.join(GT_DIR, gt_fname)

    if not os.path.exists(gt_path):
        print(f'GT tidak ditemukan untuk {fname}, dilewati')
        continue

    pred_count = predict_count(img_path)
    gt_count = load_gt_count(gt_path)
    err = pred_count - gt_count

    errors.append(err)
    results.append((fname, gt_count, pred_count, err))
    print(f'{fname:15s} | GT: {gt_count:6d} | Prediksi: {pred_count:8.2f} | Selisih: {err:8.2f}')

# --- hitung metrik akhir ---
errors = np.array(errors)
mae = np.mean(np.abs(errors))
mse = np.sqrt(np.mean(np.square(errors)))

print('\n=== Hasil Validasi ===')
print(f'Jumlah gambar dievaluasi : {len(errors)}')
print(f'MAE (Mean Absolute Error): {mae:.2f}')
print(f'MSE (Root Mean Sq. Error): {mse:.2f}')
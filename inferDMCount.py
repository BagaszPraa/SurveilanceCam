import os
import sys
import torch
import numpy as np
import cv2
import scipy.io as sio
from PIL import Image
from torchvision import transforms

# --- path setup ---
BASE_DIR = '/home/bagas/SurveilanceCam'
DM_COUNT_DIR = os.path.join(BASE_DIR, 'deps/DM_Count')
sys.path.insert(0, DM_COUNT_DIR)

from models import vgg19

# --- setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(DM_COUNT_DIR, 'pretrained_models/model_qnrf.pth')
img_path = os.path.join(BASE_DIR, 'dataset/UCF-QNRF_ECCV18/Test/img_0334.jpg')
gt_path = img_path.replace('.jpg', '_ann.mat')   # img_0334_ann.mat (format QNRF)
output_dir = os.path.join(BASE_DIR, 'video')
os.makedirs(output_dir, exist_ok=True)

print(f'Menggunakan device: {device}')

# --- load ground truth ---
def load_gt_count(mat_path):
    mat = sio.loadmat(mat_path)
    points = mat['annPoints']
    return points.shape[0]

gt_count = load_gt_count(gt_path) if os.path.exists(gt_path) else None

# --- load model ---
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- load & preprocess gambar ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path).convert('RGB')

# pastikan lebar & tinggi kelipatan 8
w, h = img.size
new_w = (w // 8) * 8
new_h = (h // 8) * 8
if (new_w, new_h) != (w, h):
    img = img.resize((new_w, new_h), Image.BILINEAR)

inputs = transform(img).unsqueeze(0).to(device)

# --- inference ---
with torch.no_grad():
    outputs, _ = model(inputs)
    count = torch.sum(outputs).item()

print(f'Perkiraan jumlah orang: {count:.2f}')
if gt_count is not None:
    selisih = count - gt_count
    print(f'Ground truth: {gt_count}')
    print(f'Selisih: {selisih:.2f}')
else:
    print('Ground truth tidak ditemukan.')

# --- buat heatmap dari density map ---
density_map = outputs[0, 0].cpu().numpy()
heatmap = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
heatmap = (heatmap * 255).astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

heatmap_path = os.path.join(output_dir, 'heatmap.png')
cv2.imwrite(heatmap_path, heatmap_color)
print(f'Heatmap disimpan di: {heatmap_path}')

# --- overlay heatmap ke gambar asli ---
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
heatmap_color = cv2.resize(heatmap_color, (img_cv.shape[1], img_cv.shape[0]))
overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

# --- tambahkan teks prediksi & ground truth di overlay ---
def draw_label(img, text, pos, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, img.shape[1] / 1500)   # skala font menyesuaikan ukuran gambar
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 5, y - th - 10), (x + tw + 5, y + 10), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

pred_text = f'Prediksi: {count:.1f}'
draw_label(overlay, pred_text, (20, 40), color=(0, 255, 255))  # kuning

if gt_count is not None:
    gt_text = f'Ground Truth: {gt_count}'
    selisih_text = f'Selisih: {count - gt_count:+.1f}  ({abs(count - gt_count) / gt_count * 100:.1f}%)'
    draw_label(overlay, gt_text, (20, 80), color=(0, 255, 0))       # hijau
    draw_label(overlay, selisih_text, (20, 120), color=(0, 165, 255))  # oranye
else:
    draw_label(overlay, 'Ground Truth: tidak ditemukan', (20, 80), color=(0, 0, 255))

overlay_path = os.path.join(output_dir, 'overlay.png')
cv2.imwrite(overlay_path, overlay)
print(f'Overlay disimpan di: {overlay_path}')
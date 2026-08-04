import os
import sys
import torch
import numpy as np
import cv2
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
img_path = os.path.join(BASE_DIR, 'video/IMG_92.jpg')
output_dir = os.path.join(BASE_DIR, 'video')
os.makedirs(output_dir, exist_ok=True)

print(f'Menggunakan device: {device}')

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

# --- buat heatmap dari density map ---
density_map = outputs[0, 0].cpu().numpy()

# normalisasi 0-255
heatmap = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
heatmap = (heatmap * 255).astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# simpan heatmap murni
heatmap_path = os.path.join(output_dir, 'heatmap.png')
cv2.imwrite(heatmap_path, heatmap_color)
print(f'Heatmap disimpan di: {heatmap_path}')

# --- overlay heatmap ke gambar asli ---
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL -> OpenCV (BGR)
heatmap_color = cv2.resize(heatmap_color, (img_cv.shape[1], img_cv.shape[0]))

overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

overlay_path = os.path.join(output_dir, 'overlay.png')
cv2.imwrite(overlay_path, overlay)
print(f'Overlay disimpan di: {overlay_path}')
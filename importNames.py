from ultralytics import YOLO
from pathlib import Path

import tensorrt
model_path = Path("models/best.pt")

# Load model
model = YOLO(str(model_path))

# Output .names dengan nama yang sama
names_path = model_path.with_suffix(".names")

with open(names_path, "w", encoding="utf-8") as f:
    for i in sorted(model.names.keys()):
        f.write(f"{model.names[i]}\n")

print(f"Saved class names to: {names_path}")
import os, random, shutil, yaml
from PIL import Image

base_dir = r"Dataset\Lane-Segmantation-dataset-original"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

#===================================== setup dataset-splited folder ==============================
NEW_DATASET_NAME = "Lane-Segmantation-Custom"
split_base = f"Dataset/{NEW_DATASET_NAME}"

# === train ===
train_img = os.path.join(split_base, "images", "train")
train_lb = os.path.join(split_base, "labels", "train")

# === val ===
val_img = os.path.join(split_base, "images", "val")
val_lb = os.path.join(split_base, "labels", "val")

for d in [train_img, val_img, train_lb, val_lb]:
    os.makedirs(d, exist_ok = True)
    
# ======================================== split ======================================
SPLIT_RATIO = 0.8
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(image_files)

split_id = int(len(image_files) * SPLIT_RATIO)
train_files = image_files[:split_id]
val_files = image_files[split_id:]

def safe_copy_image(src, dst_dir):
    name, _ = os.path.splitext(os.path.basename(src))
    dst = os.path.join(dst_dir, f"{name}.png")
    with Image.open(src) as img:
        img.convert("RGB").save(dst, "PNG")

for img in train_files:
    label = os.path.splitext(img)[0] + ".txt"
    safe_copy_image(os.path.join(images_dir, img), train_img)
    shutil.copy(os.path.join(labels_dir, label), train_lb)

for img in val_files:
    label = os.path.splitext(img)[0] + ".txt"
    safe_copy_image(os.path.join(images_dir, img), val_img)
    shutil.copy(os.path.join(labels_dir, label), val_lb)
    
print("✅ Done splitting dataset:")
print(f"Train: {len(train_files)}, Val: {len(val_files)}")

# =================== generate classes.txt ===================
# Lấy danh sách class từ các file label (nếu chưa có)
classes_file = os.path.join(split_base, "classes.txt")

all_labels = []
for f in os.listdir(base_dir):
    if f.endswith(".txt"):
        with open(os.path.join(base_dir, f), 'r', encoding="utf-8") as cl_file:
            for line in cl_file:
                if line.strip():
                    cls = line.strip().split()[0]
                    all_labels.append(cls)

# Sắp xếp và ghi file classes.txt
with open(classes_file, "w", encoding="utf-8") as f:
    for c in all_labels:
        f.write(f"{c}\n")  # đổi thành tên class thật nếu bạn có

print(f"✅ Created classes.txt with {len(all_labels)} classes")

import zipfile

# =================== zip dataset ===================
zip_path = os.path.join(os.path.dirname(split_base), "custom-dataset.zip")

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    print(f"✅ Dataset zipped at: {output_path}")

zip_folder(split_base, zip_path)

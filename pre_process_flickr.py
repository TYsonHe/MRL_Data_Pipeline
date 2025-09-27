import os
import json
import string
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ========================
# 配置路径
# ========================
# 原始标注文件
raw_annotations_path = ".\\Datasets\\Flickr30k\\results_20130124 copy.txt"
# 原始图像目录（含所有 .jpg）
raw_images_dir = ".\\Datasets\\Flickr30k\\flickr30k-images"
# 输出目录
output_dir = ".\\Datasets\\Flickr30k\\flickr30k_processed"

# 目标图像大小，可以自己设定
# (width,height)
target_size = (336, 336)

# 创建输出目录结构
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)

# ========================
# 1. 加载并解析原始标注
# ========================
print("Loading and parsing annotations...")
captions_dict = defaultdict(list)

with open(raw_annotations_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 你的文件是空格分隔，格式如: 3243094580.jpg#2 The person is...
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        img_id_with_hash, caption = parts
        image_filename = img_id_with_hash.split('#')[0]
        captions_dict[image_filename].append(caption)

captions_dict = dict(captions_dict)
print(f"Loaded {len(captions_dict)} images.")

# ========================
# 2. 文本预处理
# ========================


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())  # 清理多余空格
    return text


processed_captions = {
    img: [preprocess_text(cap) for cap in caps]
    for img, caps in captions_dict.items()
}

# ========================
# 3. 划分数据集（标准比例）
# ========================
all_images = list(processed_captions.keys())
trainval, test = train_test_split(all_images, test_size=1000, random_state=42)
train, val = train_test_split(trainval, test_size=1014, random_state=42)

splits = {
    'train': train,
    'val': val,
    'test': test
}

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# ========================
# 4. 处理并按 split 保存图像
# ========================
final_captions = {}

for split_name, img_list in splits.items():
    print(f"Processing {split_name} split...")
    for img_name in img_list:
        src_path = os.path.join(raw_images_dir, img_name)
        dst_path = os.path.join(output_dir, "images", split_name, img_name)

        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found. Skipping.")
            continue

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(dst_path)
            final_captions[img_name] = processed_captions[img_name]
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

print(f"Successfully saved {len(final_captions)} images across splits.")

# ========================
# 5. 保存标注文件
# ========================
# 全量标注
with open(os.path.join(output_dir, "annotations.json"), 'w', encoding='utf-8') as f:
    json.dump(final_captions, f, indent=2)

# 按 split 的标注（推荐用于训练/评估）
split_annotations = {}
for split_name, img_list in splits.items():
    split_annotations[split_name] = {
        img: final_captions[img] for img in img_list if img in final_captions
    }

with open(os.path.join(output_dir, "splits.json"), 'w', encoding='utf-8') as f:
    json.dump(split_annotations, f, indent=2)

print("✅ Processing complete!")
print(f"Images saved in: {os.path.join(output_dir, 'images')}")
print(f"Annotations saved in: {output_dir}/splits.json")
